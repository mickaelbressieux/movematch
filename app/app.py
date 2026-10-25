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
from functools import cache
from pathlib import Path

from flask import Flask, render_template, request, flash, redirect, url_for, g
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

import dance
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
    return yolo.ACTION_LABELS.get(move, move)


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


def run_pipeline(video_path: str, script: Script, player_names: list[str]) -> GameResult:
    """Local pose perception -> per-player timing scores -> winner, boxed from the
    winner's own pose detection at their most accurate move."""
    slots, num_people, timeline = yolo.extract_players(video_path)
    players = dance.score_players(slots, script, player_names)
    result = GameResult(script=script, players=players,
                        winner=dance.pick_winner(players),
                        num_people=num_people, timeline=timeline)

    if result.has_winner:
        winner = result.winner_player
        best = min((h for h in winner.hits if h.offset is not None),
                   key=lambda h: abs(h.offset))
        t = best.at + best.offset
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
    db = get_db()
    played_at = datetime.now(timezone.utc).isoformat()
    winner = result.winner_player
    cur = db.execute(
        """INSERT INTO games
           (played_at, challenge, winner_name, winner_number, num_people,
            timestamp, method, winner_image, script_id, winner_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        ),
    )
    db.executemany(
        """INSERT INTO scores (game_id, played_at, script_id, player_name, score)
           VALUES (?, ?, ?, ?, ?)""",
        [(cur.lastrowid, played_at, result.script.id, p.name, p.total)
         for p in result.players if p.named],
    )
    db.commit()


def process_upload(script: Script, player_names: list[str], save: bool = True):
    """Shared POST handling: validate file, transcode, run pipeline, persist.

    Returns (result, error_message). Exactly one is non-None.
    """
    video = request.files.get("video")
    if not video or video.filename == "":
        return None, "Please select a video file."
    if not allowed_file(video.filename):
        return None, f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}"

    safe_name = secure_filename(video.filename)
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    video.save(save_path)
    mp4_path = save_path

    keep = os.environ.get("KEEP_UPLOADS", "0") == "1"
    try:
        mp4_path = ensure_mp4(save_path)
        result = run_pipeline(str(mp4_path), script, player_names)
        if save:
            save_game(result)
        return result, None
    except Exception as exc:
        log.exception("Pipeline failed")
        if keep:
            log.warning("KEEP_UPLOADS=1, preserving source for inspection: %s", save_path)
        return None, f"Processing error: {exc}"
    finally:
        if not keep:
            save_path.unlink(missing_ok=True)
        if mp4_path != save_path:
            mp4_path.unlink(missing_ok=True)


def parse_names(csv: str) -> list[str]:
    return [n.strip() for n in csv.split(",") if n.strip()]


def script_payload() -> dict:
    """Scripts as the browser needs them to play: steps + labels + timing."""
    return {
        "lead": dance.AVATAR_LEAD,
        "scripts": {
            s.id: {
                "name": s.name,
                "end": dance.end_time(s),
                "steps": [{"move": st.move, "at": st.at, "label": move_label(st.move)}
                          for st in s.steps],
            }
            for s in dance.load_scripts().values()
        },
    }


def scored_page(template: str, save: bool):
    """GET shows the form; POST scores the uploaded clip against the chosen script."""
    source = request.form if request.method == "POST" else request.args
    script = dance.get_script(source.get("script")) or next(iter(dance.load_scripts().values()))
    names_csv = source.get("player_names", "")
    result: GameResult | None = None

    if request.method == "POST":
        result, error = process_upload(script, parse_names(names_csv), save=save)
        if error:
            flash(error, "error")

    return render_template(template, script=script, player_names=names_csv,
                           payload=script_payload(), result=result)


# -- Routes -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    return scored_page("index.html", save=True)


@app.route("/test", methods=["GET", "POST"])
def test_page():
    # Same scoring on an uploaded clip (its t=0 is the script's t=0); nothing saved.
    return scored_page("test.html", save=False)


@app.route("/leaderboard")
def leaderboard():
    db = get_db()
    # SQLite returns the other bare columns from the row holding MAX(score), so
    # played_at is when that best score was set.
    high_scores = {
        sid: db.execute("""
            SELECT player_name AS name, MAX(score) AS score, played_at
            FROM scores WHERE script_id = ?
            GROUP BY player_name
            ORDER BY score DESC, played_at ASC LIMIT ?
        """, (sid, HIGH_SCORES)).fetchall()
        for sid in dance.load_scripts()
    }

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


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (max {MAX_MB} MB).", "error")
    return redirect(url_for("index")), 413


if __name__ == "__main__":
    # Cross-platform entry point (works on Windows, where gunicorn doesn't).
    from waitress import serve

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"MoveMatch on http://{host}:{port}")
    serve(app, host=host, port=port, threads=4, channel_timeout=1200)
