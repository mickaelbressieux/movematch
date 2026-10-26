"""Convert the 7 hand-coded gestures into pose-library templates.

``avatars.POSES`` hold each gesture as it is drawn on its avatar: the coach the
player faces and copies like a mirror. A template is the player's own view, so
each one is ``mirror(normalize(pose))``; rendering the template's mirror gives
back exactly the avatar the old scripts showed. The legacy rules accepted either
arm, so these templates do too (``either_side``). ``wave`` stays out: the rules
could never tell it from ``raise_one_arm``.

Thresholds come from ``poses.calibrate`` against the other gestures; WEIGHTS fixes
features a still avatar exaggerates (crouch's arms are only out for balance).

    python tools/convert_gestures.py [--threshold ID=VALUE ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import avatars  # noqa: E402
import poses  # noqa: E402
from legacy_classify import ACTION_LABELS  # noqa: E402

SKIP = {"wave"}
WEIGHTS = {"crouch": {"l_upper_arm": 0.3, "l_forearm": 0.3, "r_upper_arm": 0.3,
                      "r_forearm": 0.3, "l_hand": 0.3, "r_hand": 0.3}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threshold", action="append", default=[],
                    help="per-template threshold, e.g. clap=0.85")
    args = ap.parse_args()
    thresholds = {k: float(v) for k, v in (a.split("=") for a in args.threshold)}

    made = {}
    for gid, kpts in avatars.POSES.items():
        if gid in SKIP:
            continue
        made[gid] = poses.Template(id=gid, label=ACTION_LABELS[gid],
                                   kpts=poses.mirror(poses.normalize(kpts)),
                                   either_side=True, weights=WEIGHTS.get(gid, {}),
                                   source={"kind": "legacy gesture"})
    for gid, t in made.items():
        others = [o.kpts for o in made.values() if o is not t]
        t.threshold = thresholds.get(gid) or poses.calibrate(t, others)
        print(f"{gid:14} threshold {t.threshold:.2f}  -> "
              f"{poses.save_template(t).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
