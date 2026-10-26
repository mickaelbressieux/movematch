"""Template matcher vs the legacy gesture rules, frame by frame on real clips.

The rules aren't ground truth either, so the numbers say where the two disagree
and the saved crops (tools/out/compare/) let a human decide who is right.

    python tools/compare_matcher.py VIDEO [VIDEO ...] [--crops N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tools"))
import poses  # noqa: E402
from legacy_classify import classify  # noqa: E402
from timing_proto import detect_all  # noqa: E402

OUT = ROOT / "tools" / "out" / "compare"
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]


def crop(video: Path, frame_idx: int, box, kpts, title: str, path: Path):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return
    for a, b in BONES:
        if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
            cv2.line(img, tuple(map(int, kpts[a][:2])), tuple(map(int, kpts[b][:2])), (0, 255, 0), 2)
    x0, y0, x1, y1 = map(int, box)
    pad = int(0.15 * (y1 - y0))
    h, w = img.shape[:2]
    img = img[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
    scale = 360 / max(1, img.shape[0])
    img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.putText(img, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", type=Path, nargs="+")
    ap.add_argument("--crops", type=int, default=4, help="crops saved per disagreement kind")
    args = ap.parse_args()
    lib = poses.load_library()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()

    stats = {g: {"rule": 0, "tpl": 0, "both": 0} for g in lib}
    examples = {(g, k): [] for g in lib for k in ("rule_only", "tpl_only", "both")}
    n_people = 0
    for video in args.videos:
        dets = detect_all(video)
        for j, f in enumerate(dets["fidx"]):
            raw = dets["kpts"][j]
            n = poses.normalize(raw)
            rules = classify(raw)
            n_people += 1
            for g, t in lib.items():
                sim = t.similarity(n)
                r, m = g in rules, sim >= t.threshold
                stats[g]["rule"] += r
                stats[g]["tpl"] += m
                stats[g]["both"] += r and m
                kind = "both" if r and m else "rule_only" if r else "tpl_only" if m else None
                if kind:
                    examples[(g, kind)].append((sim, video, int(f), dets["boxes"][j], raw))

    print(f"{n_people} person-frames from {len(args.videos)} clip(s)\n")
    print(f"{'gesture':16} {'rule':>6} {'tpl':>6} {'both':>6} {'jaccard':>8}")
    for g, s in stats.items():
        union = s["rule"] + s["tpl"] - s["both"]
        jac = s["both"] / union if union else float("nan")
        print(f"{g:16} {s['rule']:6d} {s['tpl']:6d} {s['both']:6d} {jac:8.2f}")

    for (g, kind), ex in examples.items():
        if not ex:
            continue
        # Spread the crops over the clip rather than taking consecutive frames.
        pick = [ex[i] for i in np.linspace(0, len(ex) - 1, min(args.crops, len(ex))).astype(int)]
        for sim, video, f, box, raw in pick:
            crop(video, f, box, raw, f"{g} {kind} sim={sim:.2f} f{f}",
                 OUT / f"{g}_{kind}_{video.stem[:12]}_{f:05d}.jpg")
    print(f"\ncrops in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
