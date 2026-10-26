"""MoveMatch perception -- local YOLO pose.

Deterministic and on-device. Every sampled frame is letterboxed to 640x640 and run
through YOLOv8n-pose (ONNX, via OpenCV's ``cv2.dnn`` on CPU). Each detection above
the confidence threshold yields a person box plus the 17 COCO keypoints in original
pixel coordinates; overlapping boxes are dropped with NMS.

Tracking is deliberately simple: the camera is static, nobody walks in, and players
mostly dance in place, so each detection joins the slot whose horizontal position
is nearest, and slots are numbered left -> right. Each slot keeps its normalized
pose per sampled frame; poses.py decides which frames match a move. The winner box
comes from the same detections, so box and timing always agree.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Callable

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
SAMPLE_FPS = 15        # frames per second actually inferred (~0.07 s timing resolution)
MATCH_DX = 0.12        # a detection joins the nearest slot within this share of the width
MIN_FRAMES = 2         # shorter tracks are noise, not players

# COCO keypoint indices.
NOSE = 0
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

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


# -- Tracking -----------------------------------------------------------------

def _extract(video_path, progress: Callable[[float], None] | None = None):
    """Sample the clip at SAMPLE_FPS, detect every person, track them left -> right.

    Returns ``(slots, num_people)``: slots ordered left -> right, each
    ``{"cx", "frames", "track": [(t, box)], "kpts": [(t, raw)],
    "poses": [(t, pose | None)], "actions": {}}`` where raw is the (17, 3) pixel
    keypoints and pose its ``poses.normalize``. ``t`` counts from the first frame.
    """
    import poses  # poses imports this module's constants

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps / SAMPLE_FPS))

    slots: list[dict] = []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        t = idx / fps
        match_thresh = MATCH_DX * frame.shape[1]

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
                slot = {"cx": cx, "frames": 0, "track": [], "kpts": [], "poses": [],
                        "actions": {}}
                slots.append(slot)
                best = len(slots) - 1
            else:
                slot = slots[best]
                slot["cx"] = 0.5 * slot["cx"] + 0.5 * cx
            used.add(best)

            slot["frames"] += 1
            slot["track"].append((t, det["box"]))
            slot["kpts"].append((t, det["kpts"]))
            slot["poses"].append((t, poses.normalize(det["kpts"])))
        if progress and total:
            progress(min(1.0, (idx + 1) / total))
    cap.release()

    # Drop slots seen too briefly to be a real player.
    slots = [s for s in slots if s["frames"] >= MIN_FRAMES]
    slots.sort(key=lambda s: s["cx"])
    return slots, len(slots)


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

def extract_players(video_path, progress: Callable[[float], None] | None = None
                    ) -> tuple[list[dict], int]:
    """Track every player in the clip; see ``_extract`` for the slot layout.
    ``progress`` gets the fraction of the clip read so far."""
    return _extract(video_path, progress)
