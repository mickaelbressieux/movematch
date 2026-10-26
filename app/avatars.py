"""MoveMatch pose avatars -- a small figure showing each move.

The avatar is the coach: a figure facing the player, drawn as the dancer would
appear on video, which the player copies like a mirror (Just Dance style). A pose
template stores the PLAYER's view, so the avatar draws its mirror image: a move
where the player raises their LEFT arm shows the coach's arm on the screen's LEFT,
the same side the player sees their own left arm in the mirrored camera preview.

``render_svg`` draws a rounded mannequin from any 17 keypoints in the 240x300
avatar frame; ``render_template`` maps a normalized template into that frame first.
``POSES`` are the hand-placed legacy gestures, the source of their templates (see
tools/convert_gestures.py). Run ``python avatars.py`` to check the pose library and
(re)generate ``static/avatars/<move>.svg`` plus a ``preview.html`` gallery.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np

import poses
from yolo import (L_ANK, L_ELB, L_HIP, L_KNE, L_SHO, L_WRI, NOSE, R_ANK,
                  R_ELB, R_HIP, R_KNE, R_SHO, R_WRI)

OUT_DIR = Path(__file__).resolve().parent / "static" / "avatars"

WIDTH, HEIGHT = 240, 300
HEAD_R = 22
TORSO_PX = 85         # torso length of a template drawn at full size
GROUND_Y = 285        # where the lowest ankle stands
MARGIN = 8            # px kept free around the figure when it must shrink to fit

# Palette: one accent body color, a darker shade for outlines / details.
BODY = "#4f7cff"
SHADE = "#2f4fb8"
SKIN = "#ffd2a8"
FX = "#ff9f1c"        # motion / effect lines

# Neutral standing pose, (x, y) per COCO keypoint. Shoulder width = 50 px.
_BASE = {
    NOSE: (120, 62),
    L_SHO: (145, 100), R_SHO: (95, 100),
    L_ELB: (155, 145), R_ELB: (85, 145),
    L_WRI: (160, 190), R_WRI: (80, 190),
    L_HIP: (135, 185), R_HIP: (105, 185),
    L_KNE: (137, 235), R_KNE: (103, 235),
    L_ANK: (138, 285), R_ANK: (102, 285),
}


def _pose(shift_y: float = 0, **joints) -> np.ndarray:
    """(17, 3) keypoints: the base pose with the upper body shifted down by
    ``shift_y`` (nose, shoulders, hips) and individual joints overridden."""
    names = {"nose": NOSE, "ls": L_SHO, "rs": R_SHO, "le": L_ELB, "re": R_ELB,
             "lw": L_WRI, "rw": R_WRI, "lh": L_HIP, "rh": R_HIP, "lk": L_KNE,
             "rk": R_KNE, "la": L_ANK, "ra": R_ANK}
    kpts = np.zeros((17, 3))
    for i, (x, y) in _BASE.items():
        moves_down = i in (NOSE, L_SHO, R_SHO, L_HIP, R_HIP)
        kpts[i] = (x, y + (shift_y if moves_down else 0), 1.0)
    for name, (x, y) in joints.items():
        kpts[names[name]] = (x, y, 1.0)
    # Eyes / ears aren't drawn, but keep them on the head so the array is a
    # plausible detection.
    for i in range(1, 5):
        kpts[i] = (kpts[NOSE][0], kpts[NOSE][1], 1.0)
    return kpts


POSES: dict[str, np.ndarray] = {
    # Forearms horizontal across the chest, wrists past each other.
    "arms_crossed": _pose(le=(172, 140), lw=(98, 142),
                          re=(68, 146), rw=(142, 148)),
    # Hands pressed together at chest height, above the elbows.
    "clap": _pose(le=(152, 158), lw=(124, 132),
                  re=(88, 158), rw=(116, 132)),
    # Elbows out, both hands resting on top of the head.
    "hands_on_head": _pose(le=(172, 72), lw=(136, 46),
                           re=(68, 72), rw=(104, 46)),
    # Right arm straight up.
    "raise_one_arm": _pose(re=(88, 58), rw=(84, 16)),
    # Right arm up, elbow out and bent: the open hand waves.
    "wave": _pose(re=(62, 92), rw=(56, 44)),
    # Hips dropped to knee height, knees out, arms out for balance.
    "crouch": _pose(shift_y=50, lk=(158, 245), rk=(82, 245),
                    le=(178, 172), lw=(204, 190),
                    re=(62, 172), rw=(36, 190)),
    # Right knee lifted, right hand reaching down onto it.
    "touch_knee": _pose(shift_y=10, rk=(100, 212), ra=(96, 258),
                        re=(78, 160), rw=(98, 204)),
    # Right arm extended straight out to the side at shoulder height.
    "point_camera": _pose(re=(62, 100), rw=(26, 100)),
}

# -- Rendering ----------------------------------------------------------------

def _pt(kpts, i) -> tuple[float, float]:
    return float(kpts[i][0]), float(kpts[i][1])


def _limb(kpts, joints, width) -> str:
    """Thick rounded limb with a dark outline, so it stays readable in front of
    the torso or another limb."""
    pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in (_pt(kpts, j) for j in joints))
    return "".join(
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        for color, w in ((SHADE, width + 5), (BODY, width)))


def _hand(kpts, i) -> str:
    x, y = _pt(kpts, i)
    return (f'<circle cx="{x:.0f}" cy="{y:.0f}" r="8" fill="{SKIN}" '
            f'stroke="{SHADE}" stroke-width="2"/>')


def _foot(kpts, i, outward: int) -> str:
    x, y = _pt(kpts, i)
    return (f'<ellipse cx="{x + 5 * outward:.0f}" cy="{y + 4:.0f}" rx="13" ry="7" '
            f'fill="{SHADE}"/>')


def _effects(action: str, kpts) -> list[str]:
    """Small motion cues for moves a still pose can't show on its own."""
    out = []
    if action == "wave":
        # "( (  hand  ) )" swish marks on both sides of the hand.
        x, y = _pt(kpts, R_WRI)
        for side in (-1, 1):
            for r in (17, 26):
                ax = x + side * r
                out.append(f'<path d="M {ax:.0f} {y - 11:.0f} Q {ax + side * 7:.0f} '
                           f'{y:.0f} {ax:.0f} {y + 11:.0f}" fill="none" stroke="{FX}" '
                           f'stroke-width="4" stroke-linecap="round"/>')
    elif action == "clap":
        (lx, ly), (rx, ry) = _pt(kpts, L_WRI), _pt(kpts, R_WRI)
        cx, cy = (lx + rx) / 2, (ly + ry) / 2
        for dx, dy in ((-0.9, -1), (0, -1.3), (0.9, -1)):
            out.append(f'<line x1="{cx + 16 * dx:.0f}" y1="{cy + 16 * dy:.0f}" '
                       f'x2="{cx + 28 * dx:.0f}" y2="{cy + 28 * dy:.0f}" '
                       f'stroke="{FX}" stroke-width="4" stroke-linecap="round"/>')
    elif action == "point_camera":
        x, y = _pt(kpts, R_WRI)
        # Index finger sticking out of the hand.
        out += [f'<line x1="{x - 7:.0f}" y1="{y:.0f}" x2="{x - 22:.0f}" y2="{y:.0f}" '
                f'stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>'
                for color, w in ((SHADE, 9), (SKIN, 5))]
    return out


def render_svg(kpts, action: str | None = None) -> str:
    """Rounded mannequin drawn from (17, 3) keypoints in the avatar frame."""
    (lsx, lsy), (rsx, rsy) = _pt(kpts, L_SHO), _pt(kpts, R_SHO)
    (lhx, lhy), (rhx, rhy) = _pt(kpts, L_HIP), _pt(kpts, R_HIP)
    nx, ny = _pt(kpts, NOSE)
    feet_y = max(kpts[L_ANK][1], kpts[R_ANK][1]) + 6

    parts = [
        # Ground shadow.
        f'<ellipse cx="{WIDTH / 2:.0f}" cy="{feet_y:.0f}" rx="52" ry="7" '
        f'fill="{SHADE}" opacity="0.15"/>',
        # Legs.
        _limb(kpts, (L_HIP, L_KNE, L_ANK), 20),
        _limb(kpts, (R_HIP, R_KNE, R_ANK), 20),
        _foot(kpts, L_ANK, +1), _foot(kpts, R_ANK, -1),
        # Torso: shoulders -> hips, corners rounded by a thick stroke (outline
        # layer first, then the body color).
        *(f'<polygon points="{rsx:.0f},{rsy:.0f} {lsx:.0f},{lsy:.0f} '
          f'{lhx + 4:.0f},{lhy:.0f} {rhx - 4:.0f},{rhy:.0f}" fill="{BODY}" '
          f'stroke="{color}" stroke-width="{w}" stroke-linejoin="round"/>'
          for color, w in ((SHADE, 21), (BODY, 16))),
        # Neck + head with a simple face.
        f'<line x1="{nx:.0f}" y1="{ny:.0f}" x2="{(lsx + rsx) / 2:.0f}" '
        f'y2="{(lsy + rsy) / 2:.0f}" stroke="{SKIN}" stroke-width="14"/>',
        f'<circle cx="{nx:.0f}" cy="{ny:.0f}" r="{HEAD_R}" fill="{SKIN}" '
        f'stroke="{SHADE}" stroke-width="2"/>',
        f'<circle cx="{nx - 8:.0f}" cy="{ny - 3:.0f}" r="2.5" fill="{SHADE}"/>',
        f'<circle cx="{nx + 8:.0f}" cy="{ny - 3:.0f}" r="2.5" fill="{SHADE}"/>',
        f'<path d="M {nx - 7:.0f} {ny + 8:.0f} Q {nx:.0f} {ny + 13:.0f} '
        f'{nx + 7:.0f} {ny + 8:.0f}" fill="none" stroke="{SHADE}" '
        f'stroke-width="2" stroke-linecap="round"/>',
        # Arms (after the head so hands can rest on it).
        _limb(kpts, (R_SHO, R_ELB, R_WRI), 16),
        _limb(kpts, (L_SHO, L_ELB, L_WRI), 16),
        _hand(kpts, R_WRI), _hand(kpts, L_WRI),
    ]
    if action:
        parts += _effects(action, kpts)
    body = "\n  ".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="{WIDTH}" height="{HEIGHT}">\n  {body}\n</svg>\n')


# -- Templates ----------------------------------------------------------------

def to_frame(n: np.ndarray) -> np.ndarray:
    """Normalized keypoints (hip-centred, torso units) -> avatar frame pixels:
    torso TORSO_PX long, hips centred, feet on the ground; shrunk if a raised arm
    or a wide stance would leave the frame."""
    n = np.asarray(n, float)
    ank = [n[i, 1] for i in (L_ANK, R_ANK) if n[i, 2] > 0]
    bottom = max(ank) if ank else 1.2  # hidden feet: assume standing
    # Each joint must stay inside the frame with its drawing around it (the head
    # circle, a hand, a foot): scale <= room / distance for every joint.
    scale = TORSO_PX
    for i in np.flatnonzero(n[:, 2] > 0):
        pad = HEAD_R + 2 if i == NOSE else 18 if i in (L_ANK, R_ANK) else MARGIN + 2
        if bottom - n[i, 1] > 1e-6:
            scale = min(scale, (GROUND_Y - pad) / (bottom - n[i, 1]))
        if abs(n[i, 0]) > 1e-6:
            scale = min(scale, (WIDTH / 2 - pad) / abs(n[i, 0]))
    hip_y = GROUND_Y - bottom * scale
    out = n.copy()
    out[:, 0] = WIDTH / 2 + n[:, 0] * scale
    out[:, 1] = hip_y + n[:, 1] * scale
    return out


def render_template(t: poses.Template) -> str:
    """The coach avatar for a template: its mirror image, mapped into the frame."""
    return render_svg(to_frame(poses.mirror(t.kpts)), t.id)


def check_mirror() -> list[str]:
    """A template where the player raises their LEFT arm must draw the raised hand
    on the screen's left half (see the module docstring)."""
    k = poses.relaxed_stance()
    k[L_ELB, :2], k[L_WRI, :2] = (0.35, -1.5), (0.4, -2.0)   # player's left arm up
    px = to_frame(poses.mirror(k))
    raised = min((L_WRI, R_WRI), key=lambda i: px[i, 1])
    ok = px[raised, 0] < WIDTH / 2
    print(f"{'ok' if ok else 'FAIL':4}  mirror: player's left arm up -> hand at "
          f"x={px[raised, 0]:.0f} of {WIDTH} (screen-{'left' if ok else 'right'})")
    return [] if ok else ["avatar shows the player's left arm on the screen's right"]


def _preview_html(lib: dict[str, poses.Template]) -> str:
    cards = []
    for t in lib.values():
        cards.append(
            f'<figure><img src="{escape(t.id)}.svg" alt="{escape(t.id)}">'
            f'<figcaption><b>{escape(t.label)}</b>'
            f'<code>{escape(t.id)} &middot; thr {t.threshold:.2f}</code></figcaption></figure>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pose avatars</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #f4f6fb; color: #1d2433;
         margin: 0; padding: 24px 16px; }}
  h1 {{ font-size: 1.3rem; margin: 0 0 16px; }}
  .grid {{ display: grid; gap: 16px;
           grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }}
  figure {{ margin: 0; background: #fff; border-radius: 12px; padding: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); text-align: center; }}
  img {{ width: 100%; max-width: 200px; height: auto; }}
  figcaption {{ display: flex; flex-direction: column; gap: 4px; font-size: .9rem; }}
  figcaption span {{ color: #5b6478; }}
  code {{ color: #8a93a6; font-size: .8rem; }}
</style></head><body>
<h1>MoveMatch pose avatars (pose library)</h1>
<div class="grid">
{chr(10).join(cards)}
</div></body></html>
"""


def main() -> int:
    errors = poses.check_library() + check_mirror()
    lib = poses.load_library()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in lib.values():
        (OUT_DIR / f"{t.id}.svg").write_text(render_template(t))
    (OUT_DIR / "preview.html").write_text(_preview_html(lib))
    print(f"Wrote {len(lib)} avatars + preview.html to {OUT_DIR}")
    if errors:
        print("\n".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
