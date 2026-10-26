"""MoveMatch choreography builder -- dance video -> playable song script.

A song project lives in ``songs_src/<sid>/``: the source video, its audio (``audio.m4a``,
aligned so audio t = 0 is the video's presentation t = 0), the YOLO tracks of every
person (``analysis.npz``) and ``meta.json`` (beats, durations, people). All times are
the video's presentation time -- what a browser's ``currentTime`` shows -- so the
review page, the saved audio and the script share one clock.

1. ``analyze`` (background job): probe stream starts, extract the audio, track the
   beat, run YOLO on everyone at yolo.SAMPLE_FPS. The slow part is person-agnostic,
   so picking another dancer or moving the sensitivity slider never re-runs it.
2. ``review(sid, slot)``: key-pose candidates for the chosen dancer. The method won
   the timing prototype (tools/timing_proto.py, Star Ac clip, 15 fps: P 0.70 /
   R 0.84 within +-0.15 s): whole-body speed minima after significant motion,
   snapped to the nearest beat or half-beat when within SNAP_TOL, a repeat of the
   same pose shortly after dropped. Each candidate's pose becomes a template
   (mirrored: the player mirrors the dancer); similar ones are grouped, and a group
   reuses a library pose when one matches.
3. ``save``: script JSON + new pose templates + their avatars + the song's audio.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

import avatars
import dance
import poses
import yolo

BASE_DIR = Path(__file__).resolve().parent
SONG_DIR = BASE_DIR / "songs_src"
AUDIO_DIR = BASE_DIR / "static" / "songs"
SCRIPT_DIR = BASE_DIR / "scripts"

# Audio analysis.
SR = 22050            # sample rate for beat tracking
N_FFT, HOP = 2048, 512
N_MELS = 40
BPM_PRIOR = 120.0     # centre of the log-normal tempo prior
BPM_RANGE = (60.0, 200.0)
TIGHTNESS = 100.0     # beat DP: how strongly beats keep the tempo

# Key-pose timing (values from the prototype's winning run).
SMOOTH_S = 0.20       # Savitzky-Golay window (at least 5 samples)
MAX_GAP_S = 0.20      # longest keypoint dropout filled by interpolation
MIN_GAP_S = 0.20      # minimum distance between two speed minima
PRE_WIN_S = 0.60      # look-back for the motion that leads into a minimum
SNAP_TOL = 0.10       # snap to a beat / half-beat this close
MERGE_GAP_S = 1.0     # a candidate repeating the previous pose this soon after...
MERGE_DIST = 0.25     # ...(mean joint distance, torso units) is dropped
MOTION_JOINTS = [0, 7, 8, 9, 10, 13, 14, 15, 16]
# Default sensitivity: speed drop relative to the median candidate's. Picked on
# the Star Ac clip (tools/eval_choreo.py); the review page's slider moves it.
DEFAULT_SENSITIVITY = 0.6

PICK_SAMPLES = 40     # frames tried when choosing the dancer-pick still
LEAD_MAX = 0.5        # s; a match starting earlier than this isn't "arriving early"


# -- Files --------------------------------------------------------------------

def project_dir(sid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", sid or ""):
        raise ValueError(f"bad song project id {sid!r}")
    return SONG_DIR / sid


def new_project(name: str, ext: str) -> tuple[str, Path]:
    """Create ``songs_src/<sid>/``; returns (sid, where to write the source video)."""
    sid = uuid.uuid4().hex
    d = project_dir(sid)
    d.mkdir(parents=True)
    _write_json(d / "meta.json", {"name": name.strip()[:60] or "Song"})
    return sid, d / f"source.{ext.lower().lstrip('.')}"


def source_video(sid: str) -> Path:
    return next(project_dir(sid).glob("source.*"))


def load_meta(sid: str) -> dict:
    return json.loads((project_dir(sid) / "meta.json").read_text())


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(path)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "song"


# -- Audio --------------------------------------------------------------------

def _ffmpeg() -> str:
    """System ffmpeg if on PATH, else the static binary bundled by imageio-ffmpeg.
    (Same lookup as app.ffmpeg_exe; app.py can't be imported from here.)"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def stream_start(video: Path, kind: str) -> float | None:
    """pts of the first packet of the first ``kind`` ('v' or 'a') stream, in s.
    ffprobe isn't bundled, so read it from ffmpeg's framecrc muxer."""
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(video), "-map", f"0:{kind}:0",
         "-c", "copy", "-frames:" + kind, "1", "-f", "framecrc", "-"],
        capture_output=True, text=True)
    tb = re.search(r"#tb \d+: (\d+)/(\d+)", proc.stdout)
    row = next((l for l in proc.stdout.splitlines() if l and not l.startswith("#")), None)
    if not tb or not row:
        return None
    return int(row.split(",")[2]) * int(tb.group(1)) / int(tb.group(2))


def decode_audio(video: Path) -> np.ndarray:
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(video), "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def extract_audio(video: Path, dst: Path, a0: float) -> None:
    """AAC copy of the soundtrack whose t = 0 is the video's presentation t = 0
    (silence is prepended when the audio stream starts late)."""
    delay = ["-af", f"adelay={int(round(a0 * 1000))}:all=1"] if a0 > 0.001 else []
    proc = subprocess.run(
        [_ffmpeg(), "-y", "-v", "error", "-i", str(video), "-vn", *delay,
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not extract the audio: {proc.stderr[-500:]}")


def _mel_filters() -> np.ndarray:
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(30), mel(SR / 2), N_MELS + 2))
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    fb = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        lo, mid, hi = edges[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return fb


def onset_envelope(y: np.ndarray) -> np.ndarray:
    """Log-mel spectral flux, one value per HOP; frame i is centred at i*HOP/SR."""
    y = np.pad(y, N_FFT // 2)
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP]
    spec = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1))
    logmel = np.log1p(1000 * spec @ _mel_filters().T)
    flux = np.concatenate([[0], np.maximum(0, np.diff(logmel, axis=0)).sum(axis=1)])
    # Remove the slow loudness trend so quiet and loud passages weigh the same.
    env = np.maximum(0, flux - np.convolve(flux, np.ones(43) / 43, mode="same"))
    return env / (env.std() or 1)


def estimate_tempo(env: np.ndarray) -> float:
    fps = SR / HOP
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    with np.errstate(divide="ignore"):
        bpm = 60 * fps / np.arange(len(ac))
    ok = (bpm >= BPM_RANGE[0]) & (bpm <= BPM_RANGE[1])
    prior = np.exp(-0.5 * (np.log2(bpm[ok] / BPM_PRIOR)) ** 2)
    return float(bpm[ok][np.argmax(ac[ok] * prior)])


def track_beats(env: np.ndarray, bpm: float) -> np.ndarray:
    """Ellis (2007) dynamic-programming beat tracker; beat times in s."""
    fps = SR / HOP
    period = 60 * fps / bpm
    win = np.exp(-0.5 * (np.arange(-period, period + 1) * 32 / period) ** 2)
    local = np.convolve(env, win, mode="same")
    cum, back = np.zeros(len(local)), np.full(len(local), -1)
    lo, hi = int(round(2 * period)), int(round(period / 2))
    for i in range(len(local)):
        prev = np.arange(max(0, i - lo), max(0, i - hi + 1))
        if len(prev):
            cost = -TIGHTNESS * np.log((i - prev) / period) ** 2
            j = np.argmax(cum[prev] + cost)
            cum[i], back[i] = local[i] + cum[prev][j] + cost[j], prev[j]
        else:
            cum[i] = local[i]
    peaks, _ = find_peaks(cum)
    if not len(peaks):
        return np.array([])
    i = peaks[cum[peaks] > 0.5 * np.median(cum[peaks])][-1]
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    beats = np.array(beats[::-1])
    strong = local[beats] > 0.1 * np.median(local[beats])  # trim silent ends
    beats = beats[np.argmax(strong):len(strong) - np.argmax(strong[::-1])]
    return beats / fps


# -- Analysis (background job) ------------------------------------------------

def _pick_still(video: Path, slots: list[dict], v0: float, out: Path) -> dict:
    """A frame showing as many tracked people as possible, saved as a JPEG for the
    dancer picker, with each slot's box there (fractions of the frame)."""
    times = sorted({round(t, 4) for s in slots for t, _ in s["track"]})
    sample = times[::max(1, len(times) // PICK_SAMPLES)] or [0.0]
    best = max(sample, key=lambda t: sum(any(abs(tt - t) < 1e-3 for tt, _ in s["track"])
                                         for s in slots))
    # Read up to the frame instead of seeking: OpenCV's time seek can land a few
    # frames off in H.264, and the boxes must match the picture.
    cap = cv2.VideoCapture(str(video))
    target = int(round(best * (cap.get(cv2.CAP_PROP_FPS) or 30.0)))
    ok = all(cap.grab() for _ in range(target + 1))
    ok, frame = cap.retrieve() if ok else (False, None)
    cap.release()
    if not ok:
        raise RuntimeError("Could not read a frame for the dancer picker.")
    h, w = frame.shape[:2]
    cv2.imwrite(str(out), frame)
    boxes = []
    for s in slots:
        b = min(s["track"], key=lambda tb: abs(tb[0] - best))[1]
        boxes.append([round(float(b[0]) / w, 4), round(float(b[1]) / h, 4),
                      round(float(b[2]) / w, 4), round(float(b[3]) / h, 4)])
    return {"t": round(v0 + best, 4), "boxes": boxes, "width": w, "height": h}


def analyze(report, sid: str) -> dict:
    """Job body: audio, beats and every person's track for a song project."""
    d = project_dir(sid)
    video = source_video(sid)
    report(0.01, "Reading the audio...")
    v0 = stream_start(video, "v") or 0.0
    a0 = stream_start(video, "a")
    beats: list[float] = []
    bpm = None
    if a0 is not None:
        extract_audio(video, d / "audio.m4a", a0)
        env = onset_envelope(decode_audio(video))
        bpm = estimate_tempo(env)
        beats = (track_beats(env, bpm) + a0).round(4).tolist()  # presentation time
    report(0.05, "Finding the dancers...")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    slots, _ = yolo.extract_players(
        video, lambda f: report(0.05 + 0.9 * f, f"Finding the dancers... {f:.0%}"))
    if not slots:
        raise RuntimeError("Nobody was detected in this video.")
    arrays = {}
    for k, s in enumerate(slots):
        arrays[f"t{k}"] = np.array([t for t, _ in s["kpts"]])
        arrays[f"kpts{k}"] = np.array([k_ for _, k_ in s["kpts"]])
        arrays[f"boxes{k}"] = np.array([b for _, b in s["track"]])
    np.savez_compressed(d / "analysis.npz", **arrays)
    report(0.97, "Saving...")
    meta = load_meta(sid)
    meta.update(v0=v0, a0=a0, bpm=bpm, beats=beats, src_fps=fps,
                step=max(1, round(fps / yolo.SAMPLE_FPS)),
                duration=v0 + nframes / fps if nframes else None,
                people=[{"cx": round(float(s["cx"]), 1), "frames": s["frames"]} for s in slots],
                pick=_pick_still(video, slots, v0, d / "pick.jpg"))
    _write_json(d / "meta.json", meta)
    return {"people": len(slots), "beats": len(beats), "bpm": bpm}


# -- Key-pose candidates ------------------------------------------------------

def _odd(n: float) -> int:
    n = max(5, int(round(n)))
    return n if n % 2 else n + 1


def dancer_series(sid: str, slot: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(times, raw (T, 17, 3)) of one person on the full sampling grid -- NaN where
    they weren't detected -- plus the grid's frame rate. Times are presentation
    times."""
    meta = load_meta(sid)
    data = np.load(project_dir(sid) / "analysis.npz")
    if f"t{slot}" not in data.files:
        raise ValueError(f"no person #{slot} in this video")
    fps, step = meta["src_fps"], meta["step"]
    t_slot, k_slot = data[f"t{slot}"], data[f"kpts{slot}"]
    n = int(round((meta["duration"] - meta["v0"]) * fps / step)) + 1 if meta["duration"] \
        else int(round(t_slot[-1] * fps / step)) + 1
    raw = np.full((n, 17, 3), np.nan)
    idx = np.clip(np.round(t_slot * fps / step).astype(int), 0, n - 1)
    raw[idx] = k_slot
    return meta["v0"] + np.arange(n) * step / fps, raw, fps / step


def _fill(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over interior NaN runs of at most ``max_gap`` samples."""
    x = x.copy()
    valid = ~np.isnan(x)
    if valid.sum() < 2:
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[valid], x[valid])
    run = np.zeros(len(x), int)
    for i in range(len(x)):
        run[i] = 0 if valid[i] else (run[i - 1] if i else 0) + 1
    for i in range(len(x) - 2, -1, -1):  # every NaN gets its whole run's length
        if run[i] and run[i + 1]:
            run[i] = run[i + 1]
    inside = (idx > idx[valid][0]) & (idx < idx[valid][-1])
    keep = ~valid & (run <= max_gap) & inside
    x[keep] = filled[keep]
    return x


def _track_normalize(raw: np.ndarray, fps: float) -> np.ndarray:
    """Hip-centred, scaled by a 1 s running median torso length (steadier than a
    per-frame scale when the dancer bends). (T, 17, 2), NaN = missing."""
    xy = raw[..., :2].copy()
    xy[~(raw[..., 2] >= yolo.KP_CONF)] = np.nan
    hip = (xy[:, yolo.L_HIP] + xy[:, yolo.R_HIP]) / 2
    sho = (xy[:, yolo.L_SHO] + xy[:, yolo.R_SHO]) / 2
    torso = np.hypot(*(sho - hip).T)
    half = int(fps / 2)
    med = np.full(len(torso), np.nan)
    for i in range(len(torso)):
        w = torso[max(0, i - half):i + half + 1]
        if np.any(~np.isnan(w)):
            med[i] = np.nanmedian(w)
    return (xy - hip[:, None]) / med[:, None, None]


def _smooth(P: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Gap-fill + Savitzky-Golay positions and velocities, NaN beyond MAX_GAP_S."""
    win = _odd(SMOOTH_S * fps)
    pos, vel = np.full_like(P, np.nan), np.full_like(P, np.nan)
    idx = np.arange(len(P))
    for j in range(17):
        for c in range(2):
            x = _fill(P[:, j, c], int(MAX_GAP_S * fps))
            miss = np.isnan(x)
            if miss.all() or len(x) < win:
                continue
            tmp = np.interp(idx, idx[~miss], x[~miss])
            p = savgol_filter(tmp, win, 2)
            v = savgol_filter(tmp, win, 2, deriv=1, delta=1 / fps)
            p[miss], v[miss] = np.nan, np.nan
            pos[:, j, c], vel[:, j, c] = p, v
    return pos, vel


def _pose_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = np.hypot(*(a[MOTION_JOINTS] - b[MOTION_JOINTS]).T)
    return float(np.nanmean(d)) if np.any(~np.isnan(d)) else np.nan


def speed_minima(speed: np.ndarray, fps: float) -> list[tuple[int, float]]:
    """(frame, speed drop) of every speed minimum that follows motion."""
    ok = ~np.isnan(speed)
    if ok.sum() < 3:
        return []
    idx = np.arange(len(speed))
    s = np.interp(idx, idx[ok], speed[ok])
    mins, _ = find_peaks(-s, distance=max(1, int(MIN_GAP_S * fps)))
    pre = int(PRE_WIN_S * fps)
    out = []
    for m in mins:
        if m >= 2:
            drop = s[max(0, m - pre):m].max() - s[m]
            if drop > 0:
                out.append((int(m), float(drop)))
    return out


def snap(t: float, grid: np.ndarray, tol: float = SNAP_TOL) -> float:
    if not len(grid):
        return t
    g = float(grid[np.argmin(np.abs(grid - t))])
    return g if abs(g - t) <= tol else t


def beat_grid(beats: list[float]) -> np.ndarray:
    b = np.asarray(beats, float)
    half = (b[:-1] + b[1:]) / 2 if len(b) > 1 else np.array([])
    return np.sort(np.concatenate([b, half]))


def candidates(sid: str, slot: int) -> list[dict]:
    """Every key-pose candidate of one dancer, in time order. ``score`` is the speed
    drop relative to the median candidate's (the sensitivity scale), ``dups`` the
    earlier candidates whose pose it repeats within MERGE_GAP_S."""
    times, raw, fps = dancer_series(sid, slot)
    pos, vel = _smooth(_track_normalize(raw, fps), fps)
    with warnings.catch_warnings():  # frames without the dancer: NaN, as intended
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmean(np.hypot(vel[:, MOTION_JOINTS, 0], vel[:, MOTION_JOINTS, 1]), axis=1)
    grid = beat_grid(load_meta(sid)["beats"])
    mins = speed_minima(speed, fps)
    med = float(np.median([d for _, d in mins])) if mins else 1.0
    out = []
    for m, drop in mins:
        out.append({"t": round(snap(float(times[m]), grid), 4), "raw_t": round(float(times[m]), 4),
                    "frame": m, "score": round(drop / med, 4)})
    for j, c in enumerate(out):
        c["dups"] = [i for i in range(j) if c["t"] - out[i]["t"] < MERGE_GAP_S
                     and _pose_dist(pos[c["frame"]], pos[out[i]["frame"]]) < MERGE_DIST]
    return out


def select(cands: list[dict], sensitivity: float) -> list[int]:
    """Indices kept at a sensitivity: score >= it, minus repeats of the previous
    kept pose. The review page runs the same rule in JS."""
    kept: list[int] = []
    for i, c in enumerate(cands):
        if c["score"] >= sensitivity and not (kept and kept[-1] in c["dups"]):
            kept.append(i)
    return kept


# -- Templates ----------------------------------------------------------------

def template_kpts(raw_frame: np.ndarray) -> np.ndarray | None:
    """The player-view template for one dancer frame: normalized like a player's
    pose (poses.normalize) and mirrored, since the player mirrors the dancer."""
    if raw_frame is None or np.isnan(raw_frame).any():
        return None
    n = poses.normalize(raw_frame)
    return None if n is None else poses.mirror(n)


def _template(tid: str, kpts, label: str = "") -> poses.Template:
    return poses.Template(id=tid, label=label or tid, kpts=kpts)


def _same(a: poses.Template, b: poses.Template) -> float:
    """Symmetric similarity: a pose is grouped only if each accepts the other."""
    return min(a.similarity(b.kpts), b.similarity(a.kpts))


def library_match(kpts) -> str | None:
    """The library pose that accepts this template (and is accepted back), if any."""
    t = _template("_", kpts)
    best, best_sim = None, 0.0
    for lib in poses.load_library().values():
        sim = min(lib.similarity(t.kpts), t.similarity(lib.kpts))
        if sim >= lib.threshold and sim > best_sim:
            best, best_sim = lib.id, sim
    return best


def group(kpts_list: list[np.ndarray | None]) -> tuple[list[str | None], dict]:
    """Group similar candidate templates. Returns (pose key per candidate, poses)
    where a key is a library id or ``new<N>``; poses map key -> {kpts, library}.
    The first member of a group (in time) is its representative."""
    keys: list[str | None] = []
    reps: dict[str, poses.Template] = {}
    out: dict[str, dict] = {}
    for kp in kpts_list:
        if kp is None:
            keys.append(None)
            continue
        t = _template("_", kp)
        best = max(reps, key=lambda k: _same(t, reps[k]), default=None)
        if best is not None and _same(t, reps[best]) >= poses.DEFAULT_THRESHOLD:
            keys.append(best)
            continue
        lib_id = library_match(kp)
        key = lib_id or f"new{sum(1 for v in out.values() if not v['library']) + 1}"
        if key not in out:
            out[key] = {"kpts": kp, "library": bool(lib_id)}
            reps[key] = poses.load_library()[lib_id] if lib_id else t
        keys.append(key)
    return keys, out


def pose_svg(kpts) -> str:
    return avatars.render_template(_template("preview", kpts))


def review(sid: str, slot: int) -> dict:
    """Everything the review page needs for one dancer."""
    meta = load_meta(sid)
    cands = candidates(sid, slot)
    times, raw, _ = dancer_series(sid, slot)
    kps = [template_kpts(raw[c["frame"]]) for c in cands]
    keys, groups = group(kps)
    lib = poses.load_library()
    for c, k, kp in zip(cands, keys, kps):
        c["pose"] = k
        c["kpts"] = None if kp is None else np.round(kp, 4).tolist()
    pose_info = {}
    for key, g in groups.items():
        pose_info[key] = {
            "label": lib[key].label if g["library"] else "",
            "library": g["library"],
            "kpts": np.round(g["kpts"], 4).tolist(),
            "svg": None if g["library"] else pose_svg(g["kpts"]),
        }
    return {"sid": sid, "slot": slot, "name": meta["name"], "beats": meta["beats"],
            "bpm": meta["bpm"], "duration": meta["duration"], "fps": meta["src_fps"],
            "sensitivity": DEFAULT_SENSITIVITY,
            "candidates": [c for c in cands if c["pose"] is not None],
            "poses": pose_info}


def pose_at(sid: str, slot: int, t: float) -> dict:
    """Template + preview for a move added by hand at presentation time ``t``:
    the dancer's nearest detected frame."""
    times, raw, _ = dancer_series(sid, slot)
    seen = np.flatnonzero(~np.isnan(raw[:, 0, 0]))
    if not len(seen):
        raise ValueError("this dancer is never detected")
    i = int(seen[np.argmin(np.abs(times[seen] - t))])
    kp = template_kpts(raw[i])
    if kp is None:
        raise ValueError("the dancer's pose isn't clear at that moment")
    lib_id = library_match(kp)
    return {"kpts": np.round(kp, 4).tolist(), "library": lib_id,
            "svg": None if lib_id else pose_svg(kp), "t": round(float(times[i]), 4)}


# -- Lead ---------------------------------------------------------------------

def step_lead(template: poses.Template, times: np.ndarray, views: list, at: float) -> float:
    """How long before ``at`` the dancer's own pose (player view) starts matching
    ``template``: the template matches while she is still arriving, so a player
    copying her exactly starts that early too. 0 when it can't be measured (no
    match near the beat, or a pose held from before LEAD_MAX)."""
    near = np.flatnonzero(np.abs(times - at) <= dance.WINDOW)
    if not len(near):
        return 0.0
    hits = [float(times[i]) for i in near if template.matches(views[i])]
    starts = [s for s, e in dance._runs(hits) if abs(s - at) <= LEAD_MAX
              and not (s <= times[near[0]] + 1e-6)]  # already matching at the edge
    if not starts:
        return 0.0
    return round(at - min(starts, key=lambda s: abs(s - at)), 3)


def member_similarity(template: poses.Template, times: np.ndarray, views: list,
                      at: float) -> float:
    """Best match of the dancer's pose to ``template`` within SNAP_TOL of ``at``
    (the move's time may have been snapped off the exact frame)."""
    near = np.flatnonzero(np.abs(times - at) <= SNAP_TOL + 1e-6)
    return max((template.similarity(views[i]) for i in near), default=0.0)


def dancer_views(sid: str, slot: int) -> tuple[np.ndarray, list]:
    """(times, player-view pose per sampled frame or None) for one dancer."""
    times, raw, _ = dancer_series(sid, slot)
    return times, [template_kpts(k) if not np.isnan(k[0, 0]) else None for k in raw]


# -- Save ---------------------------------------------------------------------

def save(sid: str, payload: dict) -> str:
    """Write the reviewed script. ``payload``: {"name", "slot", "moves": [{"t",
    "pose"}], "poses": {key: {"label", "kpts"}}} where a key is a library id or a
    new pose and slot the dancer. Returns the script id."""
    meta = load_meta(sid)
    name = str(payload.get("name") or meta["name"]).strip()[:60] or "Song"
    moves = sorted(payload.get("moves") or [], key=lambda m: float(m["t"]))
    if not moves:
        raise ValueError("the script has no moves")

    # Script id: the name's slug, unless another song already uses it.
    script_id = base = slugify(name)
    n = 2
    while (SCRIPT_DIR / f"{script_id}.json").exists():
        existing = json.loads((SCRIPT_DIR / f"{script_id}.json").read_text())
        if existing.get("source") == sid:
            break  # re-saving this song: overwrite
        script_id, n = f"{base}_{n}", n + 1

    lib = poses.load_library()
    given = payload.get("poses") or {}
    new_keys = [k for k in dict.fromkeys(m["pose"] for m in moves) if k not in lib]
    made: dict[str, poses.Template] = {}
    for i, key in enumerate(new_keys, start=1):
        info = given.get(key) or {}
        kp = np.asarray(info.get("kpts"), float)
        if kp.shape != (17, 3):
            raise ValueError(f"pose {key!r} has no keypoints")
        label = str(info.get("label") or "").strip()[:40] or f"{name} pose {i}"
        made[key] = poses.Template(id=f"{script_id}_{i}", label=label, kpts=kp,
                                   source={"kind": "video", "song": sid, "script": script_id})
    times, views = dancer_views(sid, int(payload.get("slot", 0)))
    for key, t in made.items():
        others = [o.kpts for o in lib.values()] + [o.kpts for o in made.values() if o is not t]
        t.threshold = poses.calibrate(t, others)
        # A group's moves were grouped at >= DEFAULT_THRESHOLD; calibration may ask
        # for more. Keep every move of the group matching the dancer doing it.
        own = [member_similarity(t, times, views, float(m["t"])) for m in moves
               if m["pose"] == key]
        floor = min(own, default=t.threshold) - 0.01
        t.threshold = max(poses.DEFAULT_THRESHOLD, min(t.threshold, floor))
    # Relabelled library poses keep their id; only the label changes.
    for key, info in given.items():
        label = str(info.get("label") or "").strip()[:40]
        if key in lib and label and label != lib[key].label:
            lib[key].label = label
            poses.save_template(lib[key])
    for t in made.values():
        poses.save_template(t)
        (avatars.OUT_DIR / f"{t.id}.svg").write_text(avatars.render_template(t))

    audio = None
    if (project_dir(sid) / "audio.m4a").exists():
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_dir(sid) / "audio.m4a", AUDIO_DIR / f"{script_id}.m4a")
        audio = f"songs/{script_id}.m4a"
    lib = poses.load_library()
    steps = []
    for m in moves:
        move = made[m["pose"]].id if m["pose"] in made else m["pose"]
        at = round(float(m["t"]), 3)
        step = {"move": move, "at": at}
        lead = step_lead(lib[move], times, views, at)
        if lead:
            step["lead"] = lead
        steps.append(step)
    script = {"id": script_id, "name": name, "audio": audio,
              "duration": meta.get("duration"), "source": sid, "steps": steps}
    _write_json(SCRIPT_DIR / f"{script_id}.json", script)
    return script_id
