"""Timing prototype: when does a dancer hit a key pose?

Runs several candidate detectors on one dance video and scores them against
hand-made labels (tools/label.html), so the choice of method is driven by numbers
from a real clip rather than intuition. Nothing here is used by the app yet; the
winner moves into app/ in a later step.

    python tools/timing_proto.py VIDEO --labels tools/labels/<stem>.json

Stages:

1. Audio -- decode with the bundled ffmpeg, build an onset envelope (log-mel
   spectral flux), estimate the tempo by autocorrelation and track beats with
   Ellis-style dynamic programming (no librosa: only numpy + scipy ship). Audio and
   video start times come from each stream's first packet (``-f framecrc``, since
   ffprobe isn't bundled), so beat times are expressed on the video clock.
2. Pose -- YOLO on every frame (cached), the dancer = the largest person, keypoints
   centred on the hip midpoint and scaled by torso length, short gaps filled,
   Savitzky-Golay smoothed, then joint speed.
3. Candidates -- A: speed minima after motion, B: peaks of distance from the
   neutral (median) pose, C: A/B snapped to the beat grid, D: beat-grid baseline,
   E: hold x distinctiveness with a beat bonus. Each yields (time, score); a
   threshold sweep on the score plays the role of the sensitivity knob.
4. Report -- precision / recall / F1 within +-TOL of the labels at the best
   threshold (optimistic: tuned on this clip) and with the threshold tuned on one
   half of the clip and tested on the other (honest-ish), at full fps and 15 fps.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import yolo  # noqa: E402

CACHE_DIR = ROOT / "tools" / "cache"
OUT_DIR = ROOT / "tools" / "out"

TOL = 0.15            # s, a detection within this of a label is a hit
SR = 22050            # audio sample rate for analysis
N_FFT, HOP = 2048, 512
N_MELS = 40
BPM_PRIOR = 120.0     # centre of the log-normal tempo prior
BPM_RANGE = (60.0, 200.0)
TIGHTNESS = 100.0     # beat DP: how strongly beats keep the tempo

SMOOTH_S = 0.20       # Savitzky-Golay window, s
MAX_GAP_S = 0.20      # longest keypoint dropout filled by interpolation, s
MIN_GAP_S = 0.20      # minimum distance between two candidates of one detector, s
PRE_WIN_S = 0.60      # look-back for "significant motion" before a speed minimum, s
ARRIVE = 0.30         # arrival = speed back within this fraction of the drop
REACH = 0.80          # B: arrival = this fraction of the distance peak reached
SNAP_TOLS = (0.05, 0.10, 0.15)  # C: snap to the grid when this close, s
MERGE_GAP_S = 1.0     # merge consecutive candidates closer than this...
MERGE_DIST = 0.25     # ...whose poses differ by less than this (torso units)

# Joints that carry the choreography (hips are ~0 after centring).
MOTION_JOINTS = [0, 7, 8, 9, 10, 13, 14, 15, 16]


# -- Audio --------------------------------------------------------------------

def ffmpeg_exe() -> str:
    """Same lookup as app.ffmpeg_exe (not imported: app.py opens the DB on import)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def stream_start(video: Path, kind: str) -> float | None:
    """pts of the first packet of the first ``kind`` ('v' or 'a') stream, in s."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(video), "-map", f"0:{kind}:0",
         "-c", "copy", "-frames:" + kind, "1", "-f", "framecrc", "-"],
        capture_output=True, text=True)
    tb = re.search(r"#tb \d+: (\d+)/(\d+)", proc.stdout)
    row = next((l for l in proc.stdout.splitlines() if l and not l.startswith("#")), None)
    if not tb or not row:
        return None
    pts = int(row.split(",")[2])
    return pts * int(tb.group(1)) / int(tb.group(2))


def decode_audio(video: Path) -> np.ndarray:
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32)


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
    flux = np.maximum(0, np.diff(logmel, axis=0)).sum(axis=1)
    flux = np.concatenate([[0], flux])
    # Remove the slow loudness trend so quiet and loud passages weigh the same.
    trend = np.convolve(flux, np.ones(43) / 43, mode="same")
    env = np.maximum(0, flux - trend)
    return env / (env.std() or 1)


def estimate_tempo(env: np.ndarray) -> float:
    fps = SR / HOP
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    with np.errstate(divide="ignore"):
        bpm = 60 * fps / lags
    ok = (bpm >= BPM_RANGE[0]) & (bpm <= BPM_RANGE[1])
    prior = np.exp(-0.5 * (np.log2(bpm[ok] / BPM_PRIOR)) ** 2)
    return float(bpm[ok][np.argmax(ac[ok] * prior)])


def track_beats(env: np.ndarray, bpm: float) -> np.ndarray:
    """Ellis (2007) dynamic-programming beat tracker; returns beat times in s."""
    fps = SR / HOP
    period = 60 * fps / bpm
    win = np.exp(-0.5 * (np.arange(-period, period + 1) * 32 / period) ** 2)
    local = np.convolve(env, win, mode="same")
    n = len(local)
    cum = np.zeros(n)
    back = np.full(n, -1)
    lo, hi = int(round(2 * period)), int(round(period / 2))
    for i in range(n):
        prev = np.arange(max(0, i - lo), max(0, i - hi + 1))
        if len(prev):
            cost = -TIGHTNESS * np.log((i - prev) / period) ** 2
            j = np.argmax(cum[prev] + cost)
            cum[i] = local[i] + cum[prev][j] + cost[j]
            back[i] = prev[j]
        else:
            cum[i] = local[i]
    peaks, _ = find_peaks(cum)
    if not len(peaks):
        return np.array([])
    good = peaks[cum[peaks] > 0.5 * np.median(cum[peaks])]
    i = good[-1]
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    beats = np.array(beats[::-1])
    # Drop leading / trailing beats that sit on silence.
    strong = local[beats] > 0.1 * np.median(local[beats])
    beats = beats[np.argmax(strong):len(strong) - np.argmax(strong[::-1])]
    return beats / fps


# -- Pose ---------------------------------------------------------------------

def detect_all(video: Path) -> dict:
    """YOLO on every frame, every person; cached as npz next to the tools."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{video.stem[:60]}_dets.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fidx, boxes, kpts = [], [], []
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        for det in yolo._infer(frame):
            fidx.append(i)
            boxes.append(det["box"])
            kpts.append(det["kpts"])
        if i % 100 == 0:
            print(f"  pose: frame {i}", file=sys.stderr)
    cap.release()
    out = {"fidx": np.array(fidx, int), "boxes": np.array(boxes, float).reshape(-1, 4),
           "kpts": np.array(kpts, float).reshape(-1, 17, 3),
           "fps": np.array(fps), "nframes": np.array(i + 1)}
    np.savez_compressed(cache, **out)
    return out


def dancer_track(dets: dict) -> np.ndarray:
    """(T, 17, 3) keypoints of the largest person per frame, NaN where absent."""
    T = int(dets["nframes"])
    out = np.full((T, 17, 3), np.nan)
    area = (dets["boxes"][:, 2] - dets["boxes"][:, 0]) * (dets["boxes"][:, 3] - dets["boxes"][:, 1])
    best = {}
    for j, f in enumerate(dets["fidx"]):
        if f not in best or area[j] > area[best[f]]:
            best[f] = j
    for f, j in best.items():
        out[f] = dets["kpts"][j]
    return out


def _odd(n: int) -> int:
    n = max(5, int(round(n)))
    return n if n % 2 else n + 1


def _fill(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over NaN runs of at most ``max_gap`` samples."""
    x = x.copy()
    valid = ~np.isnan(x)
    if valid.sum() < 2:
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[valid], x[valid])
    gap = np.zeros(len(x), int)
    run = 0
    for i in range(len(x)):  # length of the NaN run each NaN belongs to
        run = run + 1 if not valid[i] else 0
        gap[i] = run
    for i in range(len(x) - 2, -1, -1):
        if gap[i] and gap[i + 1]:
            gap[i] = gap[i + 1]
    edge = (idx < idx[valid][0]) | (idx > idx[valid][-1])
    keep = ~valid & (gap <= max_gap) & ~edge
    x[keep] = filled[keep]
    return x


def normalize(raw: np.ndarray, fps: float) -> np.ndarray:
    """Centre on the hip midpoint, scale by a 1 s running median torso length,
    drop low-confidence joints. Returns (T, 17, 2) with NaN for missing joints."""
    xy = raw[..., :2].copy()
    xy[raw[..., 2] < yolo.KP_CONF] = np.nan
    hip = (xy[:, yolo.L_HIP] + xy[:, yolo.R_HIP]) / 2
    sho = (xy[:, yolo.L_SHO] + xy[:, yolo.R_SHO]) / 2
    torso = np.hypot(*(sho - hip).T)
    half = int(fps / 2)
    med = np.array([np.nanmedian(torso[max(0, i - half):i + half + 1])
                    if np.any(~np.isnan(torso[max(0, i - half):i + half + 1])) else np.nan
                    for i in range(len(torso))])
    return (xy - hip[:, None]) / med[:, None, None]


def smooth(P: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Gap-fill + Savitzky-Golay. Returns (positions, velocities) of shape (T, 17, 2);
    joints missing beyond MAX_GAP_S stay NaN."""
    win = _odd(SMOOTH_S * fps)
    pos, vel = np.full_like(P, np.nan), np.full_like(P, np.nan)
    for j in range(17):
        for c in range(2):
            x = _fill(P[:, j, c], int(MAX_GAP_S * fps))
            mask = np.isnan(x)
            if mask.all() or len(x) < win:
                continue
            idx = np.arange(len(x))
            tmp = np.interp(idx, idx[~mask], x[~mask])
            p = savgol_filter(tmp, win, 2)
            v = savgol_filter(tmp, win, 2, deriv=1, delta=1 / fps)
            p[mask], v[mask] = np.nan, np.nan
            pos[:, j, c], vel[:, j, c] = p, v
    return pos, vel


def pose_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean joint distance (torso units) over motion joints visible in both."""
    d = np.hypot(*(a[MOTION_JOINTS] - b[MOTION_JOINTS]).T)
    return float(np.nanmean(d)) if np.any(~np.isnan(d)) else np.nan


# -- Candidate detectors ------------------------------------------------------

def _nanfix(x: np.ndarray) -> np.ndarray:
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    return np.interp(idx, idx[ok], x[ok]) if ok.any() else np.zeros_like(x)


def detect_A(speed, fps):
    """Speed minima after significant motion. Returns rows (t_min, t_arrive, score,
    hold_s, frame_min)."""
    s = _nanfix(speed)
    mins, _ = find_peaks(-s, distance=max(1, int(MIN_GAP_S * fps)))
    rows = []
    pre = int(PRE_WIN_S * fps)
    for m in mins:
        if m < 2:
            continue
        peak = s[max(0, m - pre):m].max()
        drop = peak - s[m]
        if drop <= 0:
            continue
        level = s[m] + ARRIVE * drop
        k = m
        while k > 0 and s[k - 1] < level:
            k -= 1
        e = m
        while e < len(s) - 1 and s[e + 1] < level:
            e += 1
        rows.append((m / fps, k / fps, drop, (e - k) / fps, m))
    return rows


def detect_B(dist, fps):
    """Peaks of distance from the neutral pose. Rows (t_peak, t_arrive, score, frame)."""
    d = _nanfix(dist)
    peaks, props = find_peaks(d, distance=max(1, int(MIN_GAP_S * fps)), prominence=1e-3)
    rows = []
    for p, prom in zip(peaks, props["prominences"]):
        level = d[p] - (1 - REACH) * prom
        k = p
        while k > 0 and d[k - 1] >= level:
            k -= 1
        rows.append((p / fps, k / fps, prom, p))
    return rows


def snap(t, grid, tol):
    if not len(grid):
        return t
    g = grid[np.argmin(np.abs(grid - t))]
    return g if abs(g - t) <= tol else t


# -- Evaluation ---------------------------------------------------------------

def match(det: np.ndarray, labels: np.ndarray, tol=TOL):
    """Greedy one-to-one matching by |error|. Returns (hits, errors)."""
    pairs = sorted((abs(d - l), i, j, d - l) for i, d in enumerate(det)
                   for j, l in enumerate(labels) if abs(d - l) <= tol)
    used_d, used_l, errs = set(), set(), []
    for _, i, j, e in pairs:
        if i in used_d or j in used_l:
            continue
        used_d.add(i)
        used_l.add(j)
        errs.append(e)
    return len(errs), errs


def prf(det, labels):
    hits, errs = match(np.asarray(det), np.asarray(labels))
    p = hits / len(det) if len(det) else 0.0
    r = hits / len(labels) if len(labels) else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, errs


def merged(cands, poses):
    """Drop a candidate that repeats the previous kept one's pose shortly after."""
    kept = []
    for c in sorted(cands, key=lambda c: c[0]):
        if kept and c[0] - kept[-1][0] < MERGE_GAP_S:
            d = pose_dist(poses[c[2]], poses[kept[-1][2]])
            if not np.isnan(d) and d < MERGE_DIST:
                continue
        kept.append(c)
    return kept


def sweep(cands, labels, poses, do_merge, t_range=None):
    """Best-F1 threshold. cands = [(t, score, frame)]. Returns (f1, p, r, th, errs, n)."""
    if t_range:
        lo, hi = t_range
        cands = [c for c in cands if lo <= c[0] < hi]
        labels = labels[(labels >= lo) & (labels < hi)]
    if not cands:
        return (0.0, 0.0, 0.0, np.inf, [], 0)
    scores = np.array([c[1] for c in cands])
    best = None
    for th in np.unique(np.quantile(scores, np.linspace(0, 0.98, 60))):
        sel = [c for c in cands if c[1] >= th]
        if do_merge:
            sel = merged(sel, poses)
        p, r, f, errs = prf([c[0] for c in sel], labels)
        if best is None or f > best[0]:
            best = (f, p, r, th, errs, len(sel))
    return best


def at_threshold(cands, labels, poses, do_merge, th, t_range):
    lo, hi = t_range
    sel = [c for c in cands if lo <= c[0] < hi and c[1] >= th]
    if do_merge:
        sel = merged(sel, poses)
    lab = labels[(labels >= lo) & (labels < hi)]
    return prf([c[0] for c in sel], lab)


# -- Main ---------------------------------------------------------------------

def build_candidates(P, fps, grid_beat, grid_half):
    """All detectors on one normalized track. Returns ({name: [(t, score, frame)]},
    smoothed poses, speed, distance-from-neutral)."""
    pos, vel = smooth(P, fps)
    speed = np.nanmean(np.hypot(vel[:, MOTION_JOINTS, 0], vel[:, MOTION_JOINTS, 1]), axis=1)
    neutral = np.nanmedian(pos, axis=0)
    dist = np.array([pose_dist(p, neutral) for p in pos])
    grid = np.sort(np.concatenate([grid_beat, grid_half]))
    out: dict[str, list] = {}

    A = detect_A(speed, fps)
    out["A speed-min (at min)"] = [(t, s, m) for t, _, s, _, m in A]
    out["A speed-min (arrival)"] = [(a, s, m) for _, a, s, _, m in A]
    B = detect_B(dist, fps)
    out["B dist-peak (at peak)"] = [(t, s, p) for t, _, s, p in B]
    out["B dist-peak (arrival)"] = [(a, s, p) for _, a, s, p in B]
    for tol in SNAP_TOLS:
        out[f"C snap A-min ±{tol:.2f}"] = [(snap(t, grid, tol), s, m) for t, _, s, _, m in A]
        out[f"C snap A-arrival ±{tol:.2f}"] = [(snap(a, grid, tol), s, m) for _, a, s, _, m in A]
        out[f"C snap B-arrival ±{tol:.2f}"] = [(snap(a, grid, tol), s, p) for _, a, s, p in B]
    for name, g in (("beats", grid_beat), ("half-beats", grid)):
        frames = np.clip(np.round(g * fps).astype(int), 0, len(pos) - 1)
        rows = []
        for i in range(1, len(frames)):
            d = pose_dist(pos[frames[i]], pos[frames[i - 1]])
            if not np.isnan(d):
                rows.append((float(g[i]), d, frames[i]))
        out[f"D grid change ({name})"] = rows
    rows = []
    for _, a, drop, hold, m in A:
        distinct = dist[m] if not np.isnan(dist[m]) else 0.0
        on_beat = len(grid) and np.min(np.abs(grid - a)) <= 0.1
        rows.append((a, (0.1 + min(hold, 1.0)) * distinct * np.sqrt(drop) * (1.5 if on_beat else 1.0), m))
    out["E hold×distinct×beat"] = rows
    return out, pos, speed, dist


def write_svg(path, dur, speed, dist, fps, beats, labels, best_name, best_det):
    W, H = 1600, 260
    x = lambda t: 20 + (W - 40) * t / dur
    def curve(v, y0, h, color):
        v = _nanfix(v)
        v = v / (np.percentile(v, 99) or 1)
        pts = " ".join(f"{x(i / fps):.1f},{y0 + h - min(1, max(0, vi)) * h:.1f}"
                       for i, vi in enumerate(v))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2"/>'
    parts = [f'<rect width="{W}" height="{H}" fill="#fff"/>',
             curve(speed, 20, 70, "#4f7cff"), curve(dist, 100, 70, "#a35bd9")]
    parts += [f'<line x1="{x(b):.1f}" x2="{x(b):.1f}" y1="180" y2="195" stroke="#999"/>' for b in beats]
    parts += [f'<line x1="{x(l):.1f}" x2="{x(l):.1f}" y1="200" y2="225" stroke="#1a9c4a" stroke-width="3"/>'
              for l in labels]
    hits_ok = {i for i, d in enumerate(best_det) if np.min(np.abs(labels - d)) <= TOL} if len(labels) else set()
    parts += [f'<circle cx="{x(d):.1f}" cy="240" r="4" fill="{"#4f7cff" if i in hits_ok else "#e5484d"}"/>'
              for i, d in enumerate(best_det)]
    for s in range(int(dur) + 1):
        parts.append(f'<text x="{x(s):.1f}" y="255" font-size="10" fill="#666">{s}</text>')
    parts.append(f'<text x="20" y="14" font-size="12" fill="#333">speed (blue), distance from '
                 f'neutral (purple), beats (grey), labels (green), {best_name} (dots: '
                 f'blue hit / red false)</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">'
                    + "".join(parts) + "</svg>")


def main() -> int:
    global SMOOTH_S
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--labels", type=Path, help="labels JSON from tools/label.html")
    ap.add_argument("--smooth", type=float, default=SMOOTH_S,
                    help=f"Savitzky-Golay window in s (default {SMOOTH_S})")
    args = ap.parse_args()
    SMOOTH_S = args.smooth

    v0 = stream_start(args.video, "v") or 0.0
    a0 = stream_start(args.video, "a")
    print(f"video start {v0:.3f} s, audio start {a0 if a0 is None else round(a0, 3)} s")
    beats = np.array([])
    if a0 is not None:
        env = onset_envelope(decode_audio(args.video))
        bpm = estimate_tempo(env)
        beats = track_beats(env, bpm) + (a0 - v0)  # on the video clock
        print(f"tempo {bpm:.1f} BPM, {len(beats)} beats, "
              f"median interval {np.median(np.diff(beats)):.3f} s")
    half = (beats[:-1] + beats[1:]) / 2 if len(beats) > 1 else np.array([])

    dets = detect_all(args.video)
    fps = float(dets["fps"])
    raw = dancer_track(dets)
    present = np.mean(~np.isnan(raw[:, 0, 0]))
    dur = len(raw) / fps
    print(f"{len(raw)} frames at {fps:.2f} fps ({dur:.1f} s), dancer found in {present:.0%}")

    labels = np.array([])
    if args.labels:
        labels = np.array(sorted(json.loads(args.labels.read_text())["labels"]), float)
        # The browser labels on presentation time (first frame at v0); here frame i
        # is at i / fps.
        labels -= v0
        print(f"{len(labels)} labels, median gap {np.median(np.diff(labels)):.2f} s, "
              f"min gap {np.min(np.diff(labels)):.2f} s")
        if len(beats):
            grid = np.sort(np.concatenate([beats, half]))
            db = np.array([np.min(np.abs(beats - l)) for l in labels])
            dh = np.array([np.min(np.abs(grid - l)) for l in labels])
            print(f"labels within 0.10 s of a beat: {np.mean(db <= .1):.0%}, "
                  f"of a beat or half-beat: {np.mean(dh <= .1):.0%} "
                  f"(median distance {np.median(db):.3f} / {np.median(dh):.3f} s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for step, label in ((1, f"{fps:.0f} fps"), (2, f"{fps / 2:.0f} fps")):
        f = fps / step
        P = normalize(raw[::step], f)
        cands, pos, speed, dist = build_candidates(P, f, beats, half)
        if not len(labels):
            for name, c in cands.items():
                print(f"  {name:28} {len(c)} candidates")
            continue
        mid = dur / 2
        print(f"\n== {label}  (TOL ±{TOL} s; 'tuned' = best threshold on the whole clip, "
              f"'split' = tuned on one half, tested on the other)")
        print(f"{'approach':30} {'merge':5} {'P':>5} {'R':>5} {'F1':>5} {'n':>4} "
              f"{'bias':>7} {'|err|':>6}  {'split F1':>8}")
        best_overall = None
        for name, c in cands.items():
            for do_merge in (False, True):
                f1, p, r, th, errs, n = sweep(c, labels, pos, do_merge)
                split = []
                for tr, te in (((0, mid), (mid, dur + 1)), ((mid, dur + 1), (0, mid))):
                    th_tr = sweep(c, labels, pos, do_merge, tr)[3]
                    split.append(at_threshold(c, labels, pos, do_merge, th_tr, te)[2])
                bias = np.median(errs) if errs else np.nan
                aerr = np.median(np.abs(errs)) if errs else np.nan
                print(f"{name:30} {'yes' if do_merge else 'no':5} {p:5.2f} {r:5.2f} {f1:5.2f} "
                      f"{n:4d} {bias:+7.3f} {aerr:6.3f}  {np.mean(split):8.2f}")
                if best_overall is None or f1 > best_overall[0]:
                    sel = [x for x in c if x[1] >= th]
                    sel = merged(sel, pos) if do_merge else sel
                    best_overall = (f1, name + (" +merge" if do_merge else ""), [x[0] for x in sel])
        svg = OUT_DIR / f"{args.video.stem[:40]}_{label.replace(' ', '')}.svg"
        write_svg(svg, dur, speed, dist, f, beats, labels, best_overall[1], np.array(best_overall[2]))
        det = np.array(best_overall[2])
        missed = [l for l in labels if not len(det) or np.min(np.abs(det - l)) > TOL]
        false = [d for d in det if np.min(np.abs(labels - d)) > TOL]
        print(f"best: {best_overall[1]}; missed labels {[round(m + v0, 2) for m in missed]}, "
              f"false detections {[round(d + v0, 2) for d in false]} (label-page time)")
        print(f"timeline: {svg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
