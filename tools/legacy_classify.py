"""The hand-coded gesture rules MoveMatch used before pose templates.

Kept out of the app as a reference: tools compare the template matcher against
these rules (agreement on real clips), and the labels name the converted legacy
templates. ``classify(kpts)`` returns the gestures visible in one person's raw
(pixel) keypoints.
"""

from __future__ import annotations

import sys
from math import hypot
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from yolo import (KP_CONF, L_ANK, L_ELB, L_HIP, L_KNE, L_SHO, L_WRI, NOSE,  # noqa: E402
                  R_ANK, R_ELB, R_HIP, R_KNE, R_SHO, R_WRI)

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


def _ok(kpts, *idxs) -> bool:
    return all(kpts[i][2] >= KP_CONF for i in idxs)


def classify(kpts) -> set[str]:
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
