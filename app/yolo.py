"""MoveMatch perception -- local YOLO pose.

Deterministic and on-device. Every sampled frame is letterboxed to 640x640 and run
through YOLOv8n-pose (ONNX, via OpenCV's ``cv2.dnn`` on CPU). Each detection above
the confidence threshold yields a person box plus the 17 COCO keypoints in original
pixel coordinates; overlapping boxes are dropped with NMS and people are tracked
left -> right by their horizontal position.

Keypoint geometry then classifies a small set of gestures per person per frame. The
result is, for every player, the times at which each gesture was seen; dance.py turns
those into timing scores against a script. The winner box comes from the same pose
detection, so box and timing always agree -- no GPU or tracker service involved.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from math import hypot
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "yolov8n-pose.onnx"
MODEL_URL = os.environ.get(
    "YOLO_POSE_URL",
    "https://huggingface.co/Xenova/yolov8n-pose/resolve/main/onnx/model.onnx",
)

INPUT_SIZE = 640
CONF_THRESH = 0.5      # person detection confidence
NMS_THRESH = 0.45
KP_CONF = 0.30         # per-keypoint confidence gate
SAMPLE_FPS = 10        # frames per second actually inferred (~0.1 s timing resolution)
MAX_FRAMES = 360       # hard cap on processed frames (CPU budget)

# COCO keypoint indices.
NOSE = 0
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

# Human-readable labels for the gestures the geometry can classify.
ACTION_LABELS = {
    "arms_crossed": "arms crossed",
    "clap": "clapping",
    "hands_on_head": "hands on head",
    "raise_one_arm": "one arm raised",
    "crouch": "crouching",
    "touch_knee": "touching a knee",
    "wave": "waving",
    "point_camera": "pointing at camera",
}

_net: cv2.dnn.Net | None = None
# cv2.dnn.Net isn't thread-safe; waitress serves requests on threads.
_net_lock = threading.Lock()


# -- Model --------------------------------------------------------------------

def _load_net() -> cv2.dnn.Net:
    """Load (downloading if needed) the pose net. Caller must hold _net_lock."""
    global _net
    if _net is None:
        if not MODEL_PATH.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            log.info("Downloading YOLOv8n-pose to %s", MODEL_PATH)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        _net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    return _net


# -- Inference ----------------------------------------------------------------

def _letterbox(img):
    """Resize keeping aspect ratio and pad to INPUT_SIZE. Returns (padded, scale,
    pad_x, pad_y) so detections can be mapped back to original pixels."""
    h, w = img.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh))
    pad_x, pad_y = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def _infer(frame) -> list[dict]:
    """Run one frame through the pose net. Returns a list of detections, each
    ``{"box": (x0, y0, x1, y1), "score": float, "kpts": ndarray(17, 3)}`` in
    original pixel coordinates."""
    padded, scale, pad_x, pad_y = _letterbox(frame)
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                 swapRB=True, crop=False)
    with _net_lock:
        net = _load_net()
        net.setInput(blob)
        out = net.forward()      # (1, 56, 8400)
    out = np.squeeze(out, 0).T   # (8400, 56): cx, cy, w, h, conf, 17*(x, y, c)

    scores = out[:, 4]
    keep = scores >= CONF_THRESH
    out, scores = out[keep], scores[keep]
    if len(out) == 0:
        return []

    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    boxes_xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
    idxs = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(),
                            CONF_THRESH, NMS_THRESH)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten()

    dets = []
    for i in idxs:
        row = out[i]
        x0 = (row[0] - row[2] / 2 - pad_x) / scale
        y0 = (row[1] - row[3] / 2 - pad_y) / scale
        x1 = (row[0] + row[2] / 2 - pad_x) / scale
        y1 = (row[1] + row[3] / 2 - pad_y) / scale
        kpts = row[5:].reshape(17, 3).copy()
        kpts[:, 0] = (kpts[:, 0] - pad_x) / scale
        kpts[:, 1] = (kpts[:, 1] - pad_y) / scale
        dets.append({"box": (x0, y0, x1, y1), "score": float(scores[i]),
                     "kpts": kpts})
    return dets


# -- Gesture classification ---------------------------------------------------

def _ok(kpts, *idxs) -> bool:
    return all(kpts[i][2] >= KP_CONF for i in idxs)


def _classify(kpts) -> set[str]:
    """Gestures visible in a single person's keypoints. Margins are normalized by
    shoulder width so the rules are scale-invariant."""
    acts: set[str] = set()
    if not _ok(kpts, L_SHO, R_SHO):
        return acts

    ls, rs = kpts[L_SHO], kpts[R_SHO]
    lw, rw = kpts[L_WRI], kpts[R_WRI]
    le, re = kpts[L_ELB], kpts[R_ELB]
    lk, rk = kpts[L_KNE], kpts[R_KNE]
    la, ra = kpts[L_ANK], kpts[R_ANK]
    lh, rh = kpts[L_HIP], kpts[R_HIP]
    nose = kpts[NOSE]

    sw = hypot(ls[0] - rs[0], ls[1] - rs[1]) or 1.0
    scx, scy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
    if _ok(kpts, L_HIP, R_HIP):
        hcx, hcy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
    else:
        hcx, hcy = scx, scy + 1.5 * sw
    head = nose if nose[2] >= KP_CONF else (scx, scy - 0.5 * sw)
    wrists = [w for w in (lw, rw) if w[2] >= KP_CONF]

    # Hands on / touching head: a wrist near the head and above the shoulders.
    for w in wrists:
        if hypot(w[0] - head[0], w[1] - head[1]) < 0.9 * sw and w[1] < scy:
            acts.add("hands_on_head")

    # Clap: both wrists close together, near chest center, raised above the elbows.
    if lw[2] >= KP_CONF and rw[2] >= KP_CONF:
        wd = hypot(lw[0] - rw[0], lw[1] - rw[1])
        midx, midy = (lw[0] + rw[0]) / 2, (lw[1] + rw[1]) / 2
        above_elbows = _ok(kpts, L_ELB, R_ELB) and lw[1] < le[1] and rw[1] < re[1]
        if (wd < 0.5 * sw and abs(midx - scx) < 0.5 * sw
                and scy < midy < hcy and above_elbows):
            acts.add("clap")

    # Arms crossed: both forearms roughly horizontal, wrists gathered at the
    # chest center but spread apart (distinguishes from a clap's tight V).
    if _ok(kpts, L_WRI, R_WRI, L_ELB, R_ELB) and "clap" not in acts:
        midy = (lw[1] + rw[1]) / 2
        horiz = abs(lw[1] - le[1]) < 0.5 * sw and abs(rw[1] - re[1]) < 0.5 * sw
        span = abs(lw[0] - rw[0])
        if (horiz and scy < midy < hcy
                and abs((lw[0] + rw[0]) / 2 - scx) < 0.6 * sw
                and 0.2 * sw < span < 1.3 * sw):
            acts.add("arms_crossed")

    # One arm raised (also counts as a wave -- best effort without motion).
    raised = 0
    if lw[2] >= KP_CONF and lw[1] < ls[1] - 0.4 * sw:
        raised += 1
    if rw[2] >= KP_CONF and rw[1] < rs[1] - 0.4 * sw:
        raised += 1
    if raised == 1:
        acts.add("raise_one_arm")
        acts.add("wave")

    # Crouch: thighs compressed -- hips drop close to the knees.
    if _ok(kpts, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK):
        thigh = ((lk[1] - lh[1]) + (rk[1] - rh[1])) / 2
        shin = ((la[1] - lk[1]) + (ra[1] - rk[1])) / 2
        if shin > 1e-3 and thigh < 0.5 * shin:
            acts.add("crouch")

    # Touching a knee: a wrist near a knee joint.
    for w in wrists:
        for k in (lk, rk):
            if k[2] >= KP_CONF and hypot(w[0] - k[0], w[1] - k[1]) < 0.7 * sw:
                acts.add("touch_knee")

    # Pointing at the camera: an arm extended roughly horizontally, far out.
    for w, s in ((lw, ls), (rw, rs)):
        if w[2] >= KP_CONF and abs(w[1] - s[1]) < 0.5 * sw and abs(w[0] - s[0]) > 1.1 * sw:
            acts.add("point_camera")

    return acts


# -- Tracking + timeline ------------------------------------------------------

def _extract(video_path):
    """Sample the clip, detect + classify per frame, and track people left->right.

    Returns ``(slots, num_people)`` where each slot is
    ``{"cx", "frames", "track": [(t, box)], "actions": {action: [t, ...]}}``,
    ordered left -> right.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps / SAMPLE_FPS))

    slots: list[dict] = []
    processed = 0
    idx = -1
    while processed < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        processed += 1
        t = idx / fps
        w = frame.shape[1]
        match_thresh = 0.12 * w

        dets = sorted(_infer(frame), key=lambda d: (d["box"][0] + d["box"][2]) / 2)
        used = set()
        for det in dets:
            x0, y0, x1, y1 = det["box"]
            cx = (x0 + x1) / 2
            best, best_d = None, match_thresh
            for si, s in enumerate(slots):
                if si in used:
                    continue
                d = abs(s["cx"] - cx)
                if d < best_d:
                    best, best_d = si, d
            if best is None:
                slot = {"cx": cx, "frames": 0, "track": [], "actions": {}}
                slots.append(slot)
                best = len(slots) - 1
            else:
                slot = slots[best]
                slot["cx"] = 0.5 * slot["cx"] + 0.5 * cx
            used.add(best)

            slot["frames"] += 1
            slot["track"].append((t, det["box"]))
            for act in _classify(det["kpts"]):
                slot["actions"].setdefault(act, []).append(t)
    cap.release()

    # Drop slots seen too briefly to be a real player.
    slots = [s for s in slots if s["frames"] >= 2]
    slots.sort(key=lambda s: s["cx"])
    return slots, len(slots)


def _build_timeline(slots) -> list[tuple]:
    """Merge each player's per-action detections into readable intervals."""
    events = []
    for rank, s in enumerate(slots, start=1):
        for act, times in s["actions"].items():
            times = sorted(times)
            start = prev = times[0]
            for t in times[1:]:
                if t - prev > 0.6:
                    events.append((start, prev, f"Person {rank}: {ACTION_LABELS[act]}"))
                    start = t
                prev = t
            events.append((start, prev, f"Person {rank}: {ACTION_LABELS[act]}"))
    return sorted(events, key=lambda e: e[0])


# -- Snapshot -----------------------------------------------------------------

def box_at(slot: dict, t: float):
    """The player's pose box from the sampled frame closest to ``t``."""
    return min(slot["track"], key=lambda tb: abs(tb[0] - t))[1] if slot["track"] else None


def winner_snapshot_yolo(video_path, timestamp, box, out_path, label="WINNER"):
    """Still frame at ``timestamp`` with the winner's pose box and ``label`` drawn."""
    cap = cv2.VideoCapture(str(video_path))
    if timestamp is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")

    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))

    color = (0, 215, 255)  # amber, BGR
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 4)
    label_y = y0 - 12 if y0 - 12 > 24 else min(h - 8, y1 + 32)
    cv2.putText(frame, label, (x0 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame)
    return out_path


# -- Orchestration ------------------------------------------------------------

def extract_players(video_path) -> tuple[list[dict], int, list[tuple]]:
    """Track and classify every player in the clip.

    Returns ``(slots, num_people, timeline)``: slots ordered left -> right, each
    ``{"cx", "frames", "track": [(t, box)], "actions": {action: [t, ...]}}``, plus a
    readable ``[(start, end, text)]`` detection timeline.
    """
    slots, num_people = _extract(video_path)
    return slots, num_people, _build_timeline(slots)
