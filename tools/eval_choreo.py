"""Score the app's key-pose proposals (choreo.py) against hand labels.

Same metric as tools/timing_proto.py -- precision / recall / F1 within +-TOL -- but
through the real pipeline (song project, analysis job, candidates, selection), so
it checks the code the review page runs and gives the default sensitivity.

    python tools/eval_choreo.py VIDEO --labels LABELS.json   # new project, analysed now
    python tools/eval_choreo.py --sid SID --labels LABELS.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tools"))
import choreo  # noqa: E402
from timing_proto import prf  # noqa: E402

SENSITIVITIES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path, nargs="?")
    ap.add_argument("--sid", help="existing song project instead of a video")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--slot", type=int, help="dancer index (default: most frames)")
    args = ap.parse_args()

    sid = args.sid
    if not sid:
        sid, dst = choreo.new_project(args.video.stem[:40], args.video.suffix)
        shutil.copyfile(args.video, dst)
        print(f"song project {sid}")
        choreo.analyze(lambda p, m=None: print(f"  {p:4.0%} {m or ''}", end="\r"), sid)
        print()
    meta = choreo.load_meta(sid)
    slot = args.slot if args.slot is not None else int(np.argmax([p["frames"] for p in meta["people"]]))
    labels = np.array(json.loads(args.labels.read_text())["labels"], float)

    data = choreo.review(sid, slot)
    cands = data["candidates"]
    print(f"{len(meta['people'])} people, dancer #{slot}; {len(meta['beats'])} beats; "
          f"{len(cands)} candidates; {len(labels)} labels (TOL ±0.15 s)")
    print(f"{'sensitivity':>11} {'P':>5} {'R':>5} {'F1':>5} {'n':>4} {'poses':>6}")
    for s in sorted(set(SENSITIVITIES) | {choreo.DEFAULT_SENSITIVITY}):
        kept = choreo.select(cands, s)
        p, r, f, _ = prf([cands[i]["t"] for i in kept], labels)
        n_poses = len({cands[i]["pose"] for i in kept})
        mark = "  <- default" if s == choreo.DEFAULT_SENSITIVITY else ""
        print(f"{s:11.2f} {p:5.2f} {r:5.2f} {f:5.2f} {len(kept):4d} {n_poses:6d}{mark}")
    lib = sorted({c["pose"] for c in cands if data["poses"][c["pose"]]["library"]})
    print(f"pose groups: {len(data['poses'])} ({len(lib)} reuse library poses: {lib})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
