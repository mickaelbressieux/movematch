"""MoveMatch -- Flask orchestrator.

Serves the dance scripts, routes the recorded clip through local YOLO pose (yolo.py),
scores every player's timing against the script (dance.py), and stores the scores
(plus a boxed winner snapshot) for the leaderboard.
"""

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from urllib.parse import quote
from functools import cache
from pathlib import Path

from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request,
                   send_file, url_for)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

import choreo
import dance
import jobs
import poses
import yolo
from models import GameResult, Script

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR = BASE_DIR / "static" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "movematch.db"

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB = 50
SONG_MAX_MB = 500  # dance videos for /create (full songs, often HD)
HIGH_SCORES = 10  # rows per script on the leaderboard

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

dance.load_scripts()  # fail at startup on an invalid script


def current_url_prefix() -> str:
    prefix = (
        request.headers.get("X-Forwarded-Prefix")
        or request.script_root
        or os.environ.get("SCRIPT_NAME", "")
    ).rstrip("/")

    if not prefix:
        host = request.headers.get("X-Forwarded-Host", request.host)
        port = request.environ.get("SERVER_PORT", "")
        # Coder workspaces (NVIDIA Launchpad) serve apps under /coder/proxy/<port>.
        if port and ".apps." in host and os.environ.get("CODER"):
            prefix = f"/coder/proxy/{port}"

    return prefix


def move_label(move: str) -> str:
    return poses.label(move)


@app.context_processor
def inject_helpers():
    return {"url_prefix": current_url_prefix(), "move_label": move_label,
            "scripts": dance.load_scripts()}


# -- Database -----------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            played_at     TEXT    NOT NULL,
            challenge     TEXT    NOT NULL,
            winner_name   TEXT,
            winner_number INTEGER DEFAULT 0,
            num_people    INTEGER DEFAULT 0,
            timestamp     REAL,
            method        TEXT,
            winner_image  TEXT,
            script_id     TEXT,
            winner_score  INTEGER
        )
    """)
    # Migrate older databases that predate these columns.
    cols = {r[1] for r in db.execute("PRAGMA table_info(games)")}
    for name, ddl in [
        ("winner_number", "INTEGER DEFAULT 0"),
        ("num_people", "INTEGER DEFAULT 0"),
        ("timestamp", "REAL"),
        ("method", "TEXT"),
        ("winner_image", "TEXT"),
        ("script_id", "TEXT"),
        ("winner_score", "INTEGER"),
        ("speed", "REAL DEFAULT 1.0"),
    ]:
        if name not in cols:
            db.execute(f"ALTER TABLE games ADD COLUMN {name} {ddl}")
    # One row per named player per game, for the per-script high scores.
    db.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id     INTEGER NOT NULL REFERENCES games(id),
            played_at   TEXT    NOT NULL,
            script_id   TEXT    NOT NULL,
            player_name TEXT    NOT NULL,
            score       INTEGER NOT NULL
        )
    """)
    # Slow practice runs (speed < 1) get their own high-score tables.
    if "speed" not in {r[1] for r in db.execute("PRAGMA table_info(scores)")}:
        db.execute("ALTER TABLE scores ADD COLUMN speed REAL DEFAULT 1.0")
    db.commit()
    db.close()


init_db()


# -- Helpers ------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@cache
def ffmpeg_exe() -> str:
    """System ffmpeg if on PATH, else the static binary bundled by imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def ensure_mp4(src: Path) -> Path:
    """Re-encode to H.264 MP4 with clean timestamps so OpenCV can read and seek it."""
    dst = src.with_name(src.stem + "_h264.mp4")
    proc = subprocess.run(
        [
            ffmpeg_exe(), "-y",
            # Browser MediaRecorder WebM has streaming headers, no seek index and
            # often no declared duration; probe generously and rebuild timestamps
            # so ffmpeg doesn't bail at container open.
            "-fflags", "+genpts+igndts",
            "-probesize", "100M",
            "-analyzeduration", "100M",
            "-i", str(src),
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-ar", "44100",
            "-movflags", "+faststart",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}) transcoding {src.name}. "
            f"The recording may have no video stream or be truncated.\n{tail}"
        )
    return dst


def run_pipeline(video_path: str, script: Script, player_names: list[str],
                 audio_offset: float = 0.0, speed: float = 1.0,
                 progress=None) -> GameResult:
    """Local pose perception -> per-player timing scores -> winner, boxed from the
    winner's own pose detection at their most accurate move. ``audio_offset`` and
    ``speed`` place the song's steps in the clip (see dance.py)."""
    slots, num_people = yolo.extract_players(video_path, progress)
    dance.detect_moves(slots, script)
    players = dance.score_players(slots, script, player_names, audio_offset, speed)
    result = GameResult(script=script, players=players,
                        winner=dance.pick_winner(players),
                        num_people=num_people, timeline=dance.build_timeline(slots),
                        speed=speed, audio_offset=audio_offset)

    if result.has_winner:
        winner = result.winner_player
        beats = dance.clip_times(script, audio_offset, speed)
        i = min((i for i, h in enumerate(winner.hits) if h.offset is not None),
                key=lambda i: abs(winner.hits[i].offset))
        t = beats[i] + winner.hits[i].offset
        box = yolo.box_at(slots[result.winner], t)
        if box:
            try:
                fname = f"{uuid.uuid4().hex}.jpg"
                yolo.winner_snapshot_yolo(video_path, t, box, SNAPSHOT_DIR / fname,
                                          label=f"WINNER {winner.total} pts")
                result.winner_image = fname
            except Exception:
                log.warning("YOLO winner snapshot failed", exc_info=True)

    return result


def save_game(result: GameResult) -> None:
    """Store a game and its named players' scores. Runs in a scoring job, outside
    any request, so it opens its own connection."""
    db = sqlite3.connect(str(DB_PATH))
    played_at = datetime.now(timezone.utc).isoformat()
    winner = result.winner_player
    cur = db.execute(
        """INSERT INTO games
           (played_at, challenge, winner_name, winner_number, num_people,
            timestamp, method, winner_image, script_id, winner_score, speed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            played_at,
            json.dumps(result.script.moves),
            winner.name if winner else None,
            winner.number if winner else 0,
            result.num_people,
            None,
            "yolo",
            result.winner_image,
            result.script.id,
            winner.total if winner else None,
            result.speed,
        ),
    )
    db.executemany(
        """INSERT INTO scores (game_id, played_at, script_id, player_name, score, speed)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(cur.lastrowid, played_at, result.script.id, p.name, p.total, result.speed)
         for p in result.players if p.named],
    )
    db.commit()
    db.close()


def score_clip(report, save_path: str, script_id: str, names: list[str], save: bool,
               audio_offset: float, speed: float, keep: bool) -> dict:
    """Job body: transcode, track, score, store. Returns ``GameResult.to_json()``."""
    save_path = Path(save_path)
    mp4_path = save_path
    script = dance.get_script(script_id)
    if script is None:
        raise ValueError(f"unknown script {script_id!r}")
    try:
        report(0.02, "Converting the clip...")
        mp4_path = ensure_mp4(save_path)
        result = run_pipeline(
            str(mp4_path), script, names, audio_offset, speed,
            lambda f: report(0.05 + 0.93 * f, f"Tracking every player... {f:.0%}"))
        if save:
            save_game(result)
        return result.to_json()
    except Exception:
        if keep:
            log.warning("KEEP_UPLOADS=1, preserving source for inspection: %s", save_path)
        raise
    finally:
        if not keep:
            save_path.unlink(missing_ok=True)
        if mp4_path != save_path:
            mp4_path.unlink(missing_ok=True)


def start_scoring(script: Script, names_csv: str, save: bool, page: str):
    """Shared POST handling: validate and store the clip, start the scoring job.

    Returns (job_id, error_message). Exactly one is non-None.
    """
    video = request.files.get("video")
    if not video or video.filename == "":
        return None, "Please select a video file."
    if not allowed_file(video.filename):
        return None, f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}"
    try:
        speed = float(request.form.get("speed") or 1.0)
        audio_offset = float(request.form.get("audio_offset") or 0.0)
    except ValueError:
        return None, "Bad speed or audio offset."
    if not 0.1 <= speed <= 2.0:
        return None, "Speed must be between 0.1 and 2."

    safe_name = secure_filename(video.filename)
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    video.save(save_path)
    keep = os.environ.get("KEEP_UPLOADS", "0") == "1"
    if keep:
        # Everything /test needs to replay the clip the same way.
        save_path.with_suffix(".json").write_text(json.dumps(
            {"script": script.id, "speed": speed, "audio_offset": audio_offset,
             "player_names": names_csv}))
    job_id = jobs.start("score", score_clip, str(save_path), script.id,
                        parse_names(names_csv), save, audio_offset, speed, keep,
                        meta={"page": page, "script": script.id,
                              "player_names": names_csv, "speed": speed})
    return job_id, None


def parse_names(csv: str) -> list[str]:
    return [n.strip() for n in csv.split(",") if n.strip()]


def script_payload() -> dict:
    """Scripts as the browser needs them to play: steps + labels + timing."""
    return {
        "lead": dance.AVATAR_LEAD,
        "speeds": dance.SPEEDS,
        "scripts": {
            s.id: {
                "name": s.name,
                "end": dance.end_time(s),
                "audio": f"{current_url_prefix()}/static/{s.audio}" if s.audio else None,
                "steps": [{"move": st.move, "at": st.at, "label": move_label(st.move)}
                          for st in s.steps],
            }
            for s in dance.load_scripts().values()
        },
    }


def scored_page(template: str, save: bool, page: str):
    """GET shows the form; POST starts scoring the uploaded clip against the chosen
    script and sends the browser to the job's result page."""
    source = request.form if request.method == "POST" else request.args
    script = dance.get_script(source.get("script")) or next(iter(dance.load_scripts().values()))
    names_csv = source.get("player_names", "")
    speed = source.get("speed", "1")

    if request.method == "POST":
        job_id, error = start_scoring(script, names_csv, save, page)
        if job_id:
            return redirect(f"{current_url_prefix()}/result/{job_id}")
        flash(error, "error")

    return render_template(template, script=script, player_names=names_csv,
                           speed=speed, payload=script_payload())


# -- Routes -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    return scored_page("index.html", save=True, page="play")


@app.route("/test", methods=["GET", "POST"])
def test_page():
    # Same scoring on an uploaded clip (placed by the given audio offset and speed);
    # nothing saved.
    return scored_page("test.html", save=False, page="test")


@app.route("/result/<job_id>")
def result_page(job_id):
    """A scoring job: progress while it runs, then the results."""
    try:
        job = jobs.get(job_id)
    except ValueError:
        job = None
    if job is None or job["kind"] != "score":
        abort(404)
    meta = job["meta"]
    result = None
    if job["status"] == "done":
        script = dance.get_script(job["result"]["script"])
        if script is None:
            abort(404)
        result = GameResult.from_json(job["result"], script)
    back = (f"/test?script={meta['script']}&speed={meta['speed']}" if meta["page"] == "test"
            else f"/?script={meta['script']}&speed={meta['speed']}"
                 f"&player_names={quote(meta['player_names'])}")
    return render_template("result.html", job=job, result=result, back_href=back,
                           back_label="New test" if meta["page"] == "test" else "Dance again",
                           player_names=meta["player_names"])


@app.route("/leaderboard")
def leaderboard():
    db = get_db()
    # SQLite returns the other bare columns from the row holding MAX(score), so
    # played_at is when that best score was set.
    # One table per script and speed: full speed always, slow practice speeds
    # only once someone has played them.
    high_scores = []
    for sid in dance.load_scripts():
        speeds = {r[0] for r in db.execute(
            "SELECT DISTINCT COALESCE(speed, 1.0) FROM scores WHERE script_id = ?", (sid,))}
        for speed in sorted(speeds | {1.0}, reverse=True):
            rows = db.execute("""
                SELECT player_name AS name, MAX(score) AS score, played_at
                FROM scores WHERE script_id = ? AND COALESCE(speed, 1.0) = ?
                GROUP BY player_name
                ORDER BY score DESC, played_at ASC LIMIT ?
            """, (sid, speed, HIGH_SCORES)).fetchall()
            high_scores.append((sid, speed, rows))

    winners = db.execute("""
        SELECT played_at, winner_name, winner_image, script_id, winner_score
        FROM games
        WHERE winner_image IS NOT NULL
        ORDER BY id DESC LIMIT 24
    """).fetchall()

    gallery = [
        {
            "played_at": w["played_at"],
            "winner_name": w["winner_name"],
            "winner_image": w["winner_image"],
            "script": dance.get_script(w["script_id"]),
            "score": w["winner_score"],
        }
        for w in winners
    ]

    return render_template("leaderboard.html", high_scores=high_scores, gallery=gallery)


# -- Script creation (dance video -> song script) ---------------------------------

def song_projects() -> list[dict]:
    """Song projects on disk, newest first, with their analysis status."""
    out = []
    for d in sorted(choreo.SONG_DIR.glob("*/meta.json"), key=lambda p: -p.stat().st_mtime):
        meta = json.loads(d.read_text())
        job = jobs.get(d.parent.name) or {}
        status = "ready" if meta.get("people") else job.get("status", "unfinished")
        out.append({"sid": d.parent.name, "name": meta.get("name", "?"),
                    "duration": meta.get("duration"), "status": status})
    return out


@app.route("/create", methods=["GET", "POST"])
def create():
    """Upload a dance video; its analysis runs as a background job."""
    if request.method == "POST":
        request.max_content_length = SONG_MAX_MB * 1024 * 1024
        video = request.files.get("video")
        if not video or not video.filename:
            flash("Please choose a dance video.", "error")
        elif not allowed_file(video.filename):
            flash(f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}", "error")
        else:
            name = request.form.get("name", "").strip() or Path(video.filename).stem
            sid, dst = choreo.new_project(name, video.filename.rsplit(".", 1)[1])
            video.save(dst)
            jobs.start("analyze", choreo.analyze, sid, job_id=sid, meta={"name": name})
            return redirect(f"{current_url_prefix()}/create/{sid}")
    return render_template("create.html", sid=None, projects=song_projects())


def _project_or_404(sid: str) -> str:
    try:
        choreo.project_dir(sid)
    except ValueError:
        abort(404)
    if not (choreo.SONG_DIR / sid / "meta.json").exists():
        abort(404)
    return sid


@app.route("/create/<sid>")
def create_review(sid):
    _project_or_404(sid)
    return render_template("create.html", sid=sid, projects=[],
                           meta=choreo.load_meta(sid), lead=dance.AVATAR_LEAD)


@app.route("/create/<sid>/video")
def create_video(sid):
    # conditional=True answers Range requests, so the review page can seek.
    return send_file(choreo.source_video(_project_or_404(sid)), conditional=True)


@app.route("/create/<sid>/pick.jpg")
def create_pick(sid):
    return send_file(choreo.project_dir(_project_or_404(sid)) / "pick.jpg")


@app.route("/api/job/<job_id>")
def job_status(job_id):
    try:
        state = jobs.get(job_id)
    except ValueError:
        state = None
    if state is None:
        abort(404)
    state.pop("trace", None)
    return jsonify(state)


@app.route("/api/song/<sid>/review")
def song_review(sid):
    _project_or_404(sid)
    try:
        return jsonify(choreo.review(sid, int(request.args.get("slot", 0))))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.route("/api/song/<sid>/pose", methods=["POST"])
def song_pose(sid):
    """Template of the dancer's pose at a time, for a move added by hand."""
    _project_or_404(sid)
    body = request.get_json(force=True)
    try:
        return jsonify(choreo.pose_at(sid, int(body["slot"]), float(body["t"])))
    except (KeyError, ValueError) as exc:
        return jsonify(error=str(exc)), 400


@app.route("/api/pose_svg", methods=["POST"])
def pose_svg():
    """Avatar preview of an unsaved template (a pose split off on the review page)."""
    return jsonify(svg=choreo.pose_svg(request.get_json(force=True)["kpts"]))


@app.route("/api/song/<sid>/save", methods=["POST"])
def song_save(sid):
    _project_or_404(sid)
    try:
        script_id = choreo.save(sid, request.get_json(force=True))
    except (KeyError, ValueError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(script=script_id, play=f"{current_url_prefix()}/?script={script_id}")


@app.errorhandler(413)
def too_large(_):
    if request.path.startswith("/create"):
        flash(f"File too large (max {SONG_MAX_MB} MB).", "error")
        return redirect(url_for("create")), 413
    flash(f"File too large (max {MAX_MB} MB).", "error")
    return redirect(url_for("index")), 413


if __name__ == "__main__":
    # Cross-platform entry point (works on Windows, where gunicorn doesn't).
    from waitress import serve

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"MoveMatch on http://{host}:{port}")
    serve(app, host=host, port=port, threads=4, channel_timeout=1200)
