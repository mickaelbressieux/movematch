"""MoveMatch pose templates -- one matcher for every move.

A move is a keypoint template: 17 COCO keypoints, normalized (centred on the hip
midpoint, scaled by torso length) and stored in the PLAYER's camera view, i.e. the
keypoints a player doing the move correctly would produce. Players mirror the
dancer (Just Dance style), so a template made from a dance video is the dancer's
pose flipped left/right (``mirror``); the avatar flips it back to show the coach.

Matching compares features that don't depend on body proportions: the angle of
each limb segment relative to the torso, plus where the hands are (clap and
hands-on-head are about hand positions more than arm angles). Each feature counts
only when its joints are visible in both poses, and segments that depart from a
relaxed stance weigh more, since they are what makes the pose. A frame is "in the
pose" when the similarity reaches the template's threshold; dance.py turns those
frames into timed runs exactly as it did for the old hand-written rules.

Templates live in ``poses/<id>.json`` (one shared library, referenced by scripts).
Run ``python poses.py`` for the self-match and confusion check.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from yolo import (KP_CONF, L_ANK, L_ELB, L_HIP, L_KNE, L_SHO, L_WRI, NOSE,
                  R_ANK, R_ELB, R_HIP, R_KNE, R_SHO, R_WRI)

POSE_DIR = Path(__file__).resolve().parent / "poses"

# COCO index of each joint's left/right counterpart.
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

SEGMENTS = {
    "l_upper_arm": (L_SHO, L_ELB), "l_forearm": (L_ELB, L_WRI),
    "r_upper_arm": (R_SHO, R_ELB), "r_forearm": (R_ELB, R_WRI),
    "l_thigh": (L_HIP, L_KNE), "l_shin": (L_KNE, L_ANK),
    "r_thigh": (R_HIP, R_KNE), "r_shin": (R_KNE, R_ANK),
}
# Joints compared by position (torso units, upright torso frame) on top of the
# angles: where the hands are tells crossed arms from a clap or a swing. Knees and
# feet are left out: their position depends on leg length vs torso, which varies
# between people (and the legacy avatars have cartoon legs).
POINTS = {"l_hand": L_WRI, "r_hand": R_WRI}
# Joint pairs whose distance defines a gesture when they touch: a clap, hands on
# the head, a hand on a knee. Each counts only when the template has that contact;
# the hand gap reaches further (GAP_REACH) because it also tells a clap from
# crossed arms.
FOLDS = ("l_fold", "r_fold")
CONTACTS = {"hand_gap": (L_WRI, R_WRI),
            "l_hand_head": (L_WRI, NOSE), "r_hand_head": (R_WRI, NOSE),
            "l_hand_l_knee": (L_WRI, L_KNE), "l_hand_r_knee": (L_WRI, R_KNE),
            "r_hand_l_knee": (R_WRI, L_KNE), "r_hand_r_knee": (R_WRI, R_KNE)}

TORSO_PER_SHOULDER = 1.7  # torso length / shoulder width, for a hidden-hips fallback
POINT_SIGMA = 0.30        # torso lengths: joint offset that scores e^-0.5
GAP_SIGMA = 0.25          # torso lengths: same for a contact distance
CONTACT_D = 0.5           # torso lengths: closer than this is a contact
GAP_REACH = 1.0           # the hand gap counts when the template's hands are this close
RELAXED_WEIGHT = 0.5      # weight of a feature where both poses are relaxed
EMPHASIS = 3.0            # up to this much extra weight where a pose departs from
                          # the relaxed stance (what makes the pose)
POINT_WEIGHT = 1.5
GAP_WEIGHT = 1.5
FOLD_WEIGHT = 1.0
FOLD_SIGMA = 0.20         # thigh/shin ratio difference that scores e^-0.5
FOLD_SPAN = 0.5           # ratio change from relaxed (1.0) counted as fully departed
TILT_WEIGHT = 1.0
MIN_COVER = 0.5           # share of a template's weight that must be visible to judge
# Set by eye on the Star Ac clip's real frames: above ~0.78 the legacy templates
# accepted only the right gesture, below it look-alikes crept in (arms up for
# hands-on-head, a hand on the belly for clap). Synthetic negatives are easier
# than real dancing, so calibrate() may raise a threshold but not lower it.
DEFAULT_THRESHOLD = 0.78
THRESHOLD_RANGE = (DEFAULT_THRESHOLD, 0.92)
# Joint noise for the self-match check, in torso lengths. YOLO's own keypoint noise
# on a held pose is ~0.006 (measured on the Star Ac clip); this models people
# doing the pose a bit differently, not the detector.
JITTER = 0.05
SELF_MATCH_MIN = 0.90     # share of jittered copies a template must still accept


# -- Normalization ------------------------------------------------------------

def normalize(kpts) -> np.ndarray | None:
    """(17, 3) pixels -> (17, 3) torso units, centred on the hip midpoint, conf 0
    for joints below KP_CONF. None when the shoulders aren't visible."""
    k = np.asarray(kpts, float)
    ok = k[:, 2] >= KP_CONF
    if not (ok[L_SHO] and ok[R_SHO]):
        return None
    sho = (k[L_SHO, :2] + k[R_SHO, :2]) / 2
    if ok[L_HIP] and ok[R_HIP]:
        hip = (k[L_HIP, :2] + k[R_HIP, :2]) / 2
    else:  # framed from the waist up: assume an upright torso
        sw = np.hypot(*(k[L_SHO, :2] - k[R_SHO, :2]))
        hip = sho + (0, TORSO_PER_SHOULDER * sw)
    torso = np.hypot(*(sho - hip))
    if torso < 1e-6:
        return None
    out = np.zeros((17, 3))
    out[:, :2] = (k[:, :2] - hip) / torso
    out[:, 2] = np.where(ok, k[:, 2], 0.0)
    return out


def mirror(n: np.ndarray) -> np.ndarray:
    """Left/right flip of a normalized pose: swap the L/R joints, negate x. Turns a
    dancer's pose into the pose of a player mirroring them, and back."""
    out = np.asarray(n, float)[FLIP].copy()
    out[:, 0] *= -1
    return out


# -- Features -----------------------------------------------------------------

@dataclass
class Features:
    angles: np.ndarray    # (8,) SEGMENTS angles, 0 = hanging straight down
    ang_vis: np.ndarray   # (8,) bool
    tilt: float           # torso lean vs image vertical
    points: np.ndarray    # (2, 2) POINTS in the upright torso frame
    pt_vis: np.ndarray    # (2,) bool
    folds: np.ndarray     # (2,) thigh / shin length per leg (L, R), NaN if hidden
    contacts: np.ndarray  # (7,) CONTACTS distances, NaN where a joint is hidden


def features(n: np.ndarray) -> Features:
    """Rotate the pose so the torso is upright (shoulders at (0, -1)) and measure
    every segment there, so a lean doesn't change the limb angles."""
    vis = n[:, 2] > 0
    sho = (n[L_SHO, :2] + n[R_SHO, :2]) / 2   # hip midpoint is the origin
    tilt = float(np.arctan2(sho[0], -sho[1]))
    c, s = np.cos(-tilt), np.sin(-tilt)
    xy = n[:, :2] @ np.array([[c, s], [-s, c]])
    angles, ang_vis = np.zeros(len(SEGMENTS)), np.zeros(len(SEGMENTS), bool)
    for i, (a, b) in enumerate(SEGMENTS.values()):
        d = xy[b] - xy[a]
        angles[i] = np.arctan2(d[0], d[1])  # from +y (down) towards +x
        ang_vis[i] = vis[a] and vis[b]
    idx = list(POINTS.values())
    # A thigh coming towards the camera (crouch, knee lift) looks short next to its
    # shin; the ratio doesn't depend on how long someone's legs are.
    folds = np.full(2, np.nan)
    for i, (h, k, a) in enumerate(((L_HIP, L_KNE, L_ANK), (R_HIP, R_KNE, R_ANK))):
        shin = np.hypot(*(xy[a] - xy[k]))
        if vis[h] and vis[k] and vis[a] and shin > 1e-6:
            folds[i] = min(2.0, np.hypot(*(xy[k] - xy[h])) / shin)
    contacts = np.array([np.hypot(*(xy[a] - xy[b])) if vis[a] and vis[b] else np.nan
                         for a, b in CONTACTS.values()])
    return Features(angles, ang_vis, tilt, xy[idx], vis[idx], folds, contacts)


def relaxed_stance() -> np.ndarray:
    """Standing, arms down: the reference for "what makes a pose", and a pose no
    template may accept."""
    k = np.zeros((17, 3))
    pts = {0: (0, -1.45), L_SHO: (0.3, -1), R_SHO: (-0.3, -1), L_ELB: (0.38, -0.47),
           R_ELB: (-0.38, -0.47), L_WRI: (0.42, 0.05), R_WRI: (-0.42, 0.05),
           L_HIP: (0.18, 0), R_HIP: (-0.18, 0), L_KNE: (0.19, 0.59), R_KNE: (-0.19, 0.59),
           L_ANK: (0.2, 1.18), R_ANK: (-0.2, 1.18)}
    for i in range(17):
        k[i] = (*pts.get(i, pts[0]), 1.0)
    return k


_RELAXED = features(relaxed_stance())


def _wrap(a):
    return np.angle(np.exp(1j * np.asarray(a)))


def departures(f: Features) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """How far each feature is from the relaxed stance, 0..1: (segments, points,
    folds, tilt). NaN folds count as relaxed."""
    seg = np.abs(_wrap(f.angles - _RELAXED.angles)) / np.pi
    pts = np.minimum(1, np.hypot(*(f.points - _RELAXED.points).T))
    folds = np.nan_to_num(np.minimum(1, np.abs(f.folds - 1) / FOLD_SPAN))
    return seg, pts, folds, min(1.0, abs(f.tilt) / (np.pi / 4))


def contact_weights(f: Features) -> np.ndarray:
    """Weight of each CONTACTS pair for this template: 0 unless it is a contact,
    more the tighter it is."""
    w = np.zeros(len(CONTACTS))
    for i, (k, d) in enumerate(zip(CONTACTS, f.contacts)):
        reach = GAP_REACH if k == "hand_gap" else CONTACT_D
        if not np.isnan(d) and d < reach:
            w[i] = GAP_WEIGHT * (RELAXED_WEIGHT + EMPHASIS * (1 - d / reach))
    return w


def _override(names, over: dict) -> np.ndarray:
    return np.array([over.get(k, np.nan) for k in names], float)


# -- Templates ----------------------------------------------------------------

@dataclass
class Template:
    id: str
    label: str
    kpts: np.ndarray                     # (17, 3) normalized, player view
    threshold: float = DEFAULT_THRESHOLD
    either_side: bool = False            # also accept the mirror image
    weights: dict = field(default_factory=dict)  # feature name -> fixed weight
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        self.kpts = np.asarray(self.kpts, float)
        views = [self.kpts] + ([mirror(self.kpts)] if self.either_side else [])
        self._sides = []
        for v in views:
            f = features(v)
            con = contact_weights(f)
            con_over = _override(CONTACTS, self.weights)
            con = np.where(np.isnan(con_over), con, con_over)
            self._sides.append((f, departures(f), con))
        self._over = (_override(SEGMENTS, self.weights), _override(POINTS, self.weights),
                      _override(FOLDS, self.weights), self.weights.get("tilt", np.nan))

    def similarity(self, n: np.ndarray | None) -> float:
        """0..1 match of a normalized pose (best over the accepted sides)."""
        if n is None:
            return 0.0
        p = features(n)
        p_dep = departures(p)
        return max(_similarity(p, p_dep, t, t_dep, con, self._over)
                   for t, t_dep, con in self._sides)

    def matches(self, n: np.ndarray | None) -> bool:
        return self.similarity(n) >= self.threshold

    def to_json(self) -> dict:
        return {"id": self.id, "label": self.label,
                "kpts": np.round(self.kpts, 4).tolist(),
                "threshold": round(self.threshold, 3), "either_side": self.either_side,
                "weights": self.weights, "source": self.source}


def _weights(p_dep, t_dep, scale, over):
    """A feature weighs more where EITHER pose departs from the relaxed stance:
    the template's raised arm must be matched, and so must the player's extra
    one (two arms up is not "raise one arm")."""
    w = scale * (RELAXED_WEIGHT + EMPHASIS * np.maximum(p_dep, t_dep))
    return np.where(np.isnan(over), w, over)


def _similarity(p: Features, p_dep, t: Features, t_dep, con_w, over) -> float:
    """Weighted mean of per-feature scores over the features visible in both poses;
    0 when too little of the template's weight is visible to judge."""
    seg_over, pt_over, fold_over, tilt_over = over
    # Torso lean (always visible: the shoulders are required).
    w = float(_weights(p_dep[3], t_dep[3], TILT_WEIGHT, np.array(tilt_over)))
    num, den, total = w * (1 + np.cos(p.tilt - t.tilt)) / 2, w, w
    # Limb angles.
    seg_w = _weights(p_dep[0], t_dep[0], 1.0, seg_over)
    both = p.ang_vis & t.ang_vis
    total += seg_w[t.ang_vis].sum()
    if both.any():
        num += (seg_w[both] * (1 + np.cos(p.angles[both] - t.angles[both])) / 2).sum()
        den += seg_w[both].sum()
    # Hand positions.
    pt_w = _weights(p_dep[1], t_dep[1], POINT_WEIGHT, pt_over)
    both = p.pt_vis & t.pt_vis
    total += pt_w[t.pt_vis].sum()
    if both.any():
        d2 = ((p.points[both] - t.points[both]) ** 2).sum(axis=1)
        num += (pt_w[both] * np.exp(-d2 / (2 * POINT_SIGMA ** 2))).sum()
        den += pt_w[both].sum()
    # Leg folds.
    fold_w = _weights(p_dep[2], t_dep[2], FOLD_WEIGHT, fold_over)
    t_vis, both = ~np.isnan(t.folds), ~np.isnan(t.folds) & ~np.isnan(p.folds)
    total += fold_w[t_vis].sum()
    if both.any():
        num += (fold_w[both] * np.exp(-(p.folds[both] - t.folds[both]) ** 2
                                      / (2 * FOLD_SIGMA ** 2))).sum()
        den += fold_w[both].sum()
    # Contacts. Closer than the template is fine, except for the hand gap: hands
    # pressed together are not crossed arms.
    used = con_w > 0
    total += con_w[used].sum()
    idx = np.flatnonzero(used & ~np.isnan(p.contacts))
    if len(idx):
        diff = p.contacts[idx] - t.contacts[idx]
        diff[idx > 0] = np.maximum(diff[idx > 0], 0)  # idx 0 is the hand gap
        num += (con_w[idx] * np.exp(-diff ** 2 / (2 * GAP_SIGMA ** 2))).sum()
        den += con_w[idx].sum()
    if den < MIN_COVER * total:
        return 0.0
    return float(num / den)


def load_template(path: Path) -> Template:
    d = json.loads(path.read_text())
    return Template(id=str(d["id"]), label=str(d.get("label", d["id"])),
                    kpts=np.array(d["kpts"], float),
                    threshold=float(d.get("threshold", DEFAULT_THRESHOLD)),
                    either_side=bool(d.get("either_side", False)),
                    weights=dict(d.get("weights", {})), source=dict(d.get("source", {})))


def save_template(t: Template) -> Path:
    POSE_DIR.mkdir(exist_ok=True)
    path = POSE_DIR / f"{t.id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(t.to_json(), indent=1) + "\n")
    tmp.replace(path)  # atomic, so another worker never reads half a file
    return path


_library: tuple[tuple, dict[str, Template]] = ((), {})


def load_library() -> dict[str, Template]:
    """All templates by id. Reloaded when a file changes, so a pose saved by one
    server worker is seen by the others."""
    global _library
    paths = sorted(POSE_DIR.glob("*.json"))
    stamp = tuple((p.name, p.stat().st_mtime_ns) for p in paths)
    if stamp != _library[0]:
        _library = (stamp, {t.id: t for t in map(load_template, paths)})
    return _library[1]


def label(move: str) -> str:
    t = load_library().get(move)
    return t.label if t else move


def match_times(frames: list[tuple[float, np.ndarray | None]], template: Template) -> list[float]:
    """Times of the frames whose pose matches ``template``."""
    return [t for t, n in frames if template.matches(n)]


# -- Self-check ---------------------------------------------------------------

def calibrate(t: Template, negatives: list[np.ndarray], seed: int = 0) -> float:
    """Threshold halfway between the closest negative pose (other templates, the
    relaxed stance) and the 5th percentile of the template's jittered copies, so
    the template keeps accepting sloppy versions of itself without accepting its
    neighbours."""
    rng = np.random.default_rng(seed)
    noisy = []
    for _ in range(400):
        k = t.kpts.copy()
        k[:, :2] += rng.normal(0, JITTER, (17, 2))
        noisy.append(t.similarity(k))
    neg = max([t.similarity(n) for n in negatives] + [t.similarity(relaxed_stance())])
    mid = (neg + float(np.percentile(noisy, 5))) / 2
    return float(np.clip(mid, *THRESHOLD_RANGE))


def check_library(lib: dict[str, Template] | None = None, seed: int = 0,
                  verbose: bool = True) -> list[str]:
    """Self-match on jittered copies, confusion between templates, relaxed stance.
    Returns the problems found (empty = ok)."""
    lib = load_library() if lib is None else lib
    rng = np.random.default_rng(seed)
    ids = list(lib)
    problems = []
    if verbose:
        print(f"{'template':16} {'thr':>5} {'self':>5}  most similar other (sim)")
    for a in ids:
        t = lib[a]
        noisy = []
        for _ in range(200):
            k = t.kpts.copy()
            k[:, :2] += rng.normal(0, JITTER, (17, 2))
            noisy.append(t.similarity(k))
        self_rate = np.mean(np.array(noisy) >= t.threshold)
        others = [(lib[b].similarity(t.kpts), b) for b in ids if b != a]
        others += [(t.similarity(relaxed_stance()), "(relaxed stance)")]
        worst = max(others)
        if verbose:
            print(f"{a:16} {t.threshold:5.2f} {self_rate:5.0%}  {worst[1]} ({worst[0]:.2f})")
        if self_rate < SELF_MATCH_MIN:
            problems.append(f"{a}: accepts only {self_rate:.0%} of jittered copies")
        for sim, b in others:
            owner = t if b == "(relaxed stance)" else lib[b]
            if sim >= owner.threshold:
                problems.append(f"{b} accepts {a} ({sim:.2f} >= {owner.threshold:.2f})"
                                if b != "(relaxed stance)" else
                                f"{a} accepts the relaxed stance ({sim:.2f})")
    return problems


def confusion(lib: dict[str, Template] | None = None) -> tuple[list[str], np.ndarray]:
    """sim[i, j] = template j's similarity to template i's pose."""
    lib = load_library() if lib is None else lib
    ids = list(lib)
    return ids, np.array([[lib[b].similarity(lib[a].kpts) for b in ids] for a in ids])


if __name__ == "__main__":
    issues = check_library()
    ids, m = confusion()
    print("\nconfusion (row pose vs column template):")
    print(" " * 16 + "".join(f"{i[:7]:>8}" for i in ids))
    for i, row in zip(ids, m):
        print(f"{i:16}" + "".join(f"{v:8.2f}" for v in row))
    print("\n" + ("\n".join(issues) if issues else "all checks passed"))
    sys.exit(1 if issues else 0)
