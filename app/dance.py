"""MoveMatch dance scripts and timing scores.

A script (``scripts/<id>.json``) is a list of timed steps::

    {"id": "warmup", "name": "Warm-up",
     "steps": [{"move": "clap", "at": 2.0}, {"move": "raise_one_arm", "at": 4.0}]}

``move`` is a YOLO action id (see ``yolo.ACTION_LABELS``) and ``at`` the beat time in
seconds from the start of the recording. While dancing, each move's avatar slides in
``AVATAR_LEAD`` seconds before its beat.

Scoring works per player and per step. Pose detections of the step's move are merged
into runs (a gap shorter than ``DEBOUNCE`` doesn't end a run), and the first run that
STARTS within ``WINDOW`` of the beat gives the player's offset. A run already going
when the window opens counts as the earliest allowed start (-WINDOW). The offset then
maps to a rating tier.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import yolo
from models import MoveHit, PlayerScore, Script, Step

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"

# The pose classifier can't tell a wave from a raised arm, so scripts use the latter.
SCRIPT_MOVES = set(yolo.ACTION_LABELS) - {"wave"}

AVATAR_LEAD = 2.0   # s an avatar takes to slide from the right edge to the hit line
WINDOW = 1.0        # s before / after the beat in which a move's start counts
DEBOUNCE = 0.3      # s a pose must be absent before reappearing counts as a new start
END_GRACE = 1.0     # s recorded after the last beat

# (max |offset| in s, rating, points), best first. Beyond the last tier: a miss.
TIERS = [
    (0.2, "Perfect", 100),
    (0.4, "Great", 70),
    (0.7, "Good", 40),
    (1.0, "OK", 10),
]
MISS = ("Miss", 0)
_EPS = 1e-6  # sample times are idx / fps floats


# -- Scripts ------------------------------------------------------------------

def _parse(path: Path) -> Script:
    data = json.loads(path.read_text())
    steps = tuple(Step(str(s["move"]), float(s["at"])) for s in data["steps"])
    if not steps:
        raise ValueError(f"{path.name}: no steps")
    for prev, step in zip((None, *steps), steps):
        if step.move not in SCRIPT_MOVES:
            raise ValueError(f"{path.name}: unknown move {step.move!r} "
                             f"(allowed: {', '.join(sorted(SCRIPT_MOVES))})")
        if prev and step.at <= prev.at:
            raise ValueError(f"{path.name}: step times must increase ({step.at})")
        if prev and step.move == prev.move:
            raise ValueError(f"{path.name}: {step.move!r} twice in a row at {step.at}")
    return Script(id=str(data["id"]), name=str(data["name"]), steps=steps)


@cache
def load_scripts() -> dict[str, Script]:
    """All scripts by id, shortest first. Raises on an invalid file, or when two
    scripts share an id or a move count (one script per count)."""
    scripts: dict[str, Script] = {}
    counts: dict[int, str] = {}
    for path in sorted(SCRIPT_DIR.glob("*.json")):
        script = _parse(path)
        n = len(script.steps)
        if script.id in scripts:
            raise ValueError(f"{path.name}: duplicate script id {script.id!r}")
        if n in counts:
            raise ValueError(f"{path.name}: {counts[n]!r} already has {n} moves")
        scripts[script.id] = script
        counts[n] = script.id
    return dict(sorted(scripts.items(), key=lambda kv: len(kv[1].steps)))


def get_script(script_id: str | None) -> Script | None:
    return load_scripts().get(script_id or "")


def end_time(script: Script) -> float:
    """When recording stops for this script."""
    return script.steps[-1].at + END_GRACE


# -- Scoring ------------------------------------------------------------------

def _runs(times: list[float]) -> list[tuple[float, float]]:
    """Merge sorted detection times into (start, end) runs of a held pose."""
    runs: list[tuple[float, float]] = []
    for t in times:
        if runs and t - runs[-1][1] < DEBOUNCE:
            runs[-1] = (runs[-1][0], t)
        else:
            runs.append((t, t))
    return runs


def step_offset(runs: list[tuple[float, float]], beat: float) -> float | None:
    """Signed start offset vs ``beat`` from a move's runs, or None if missed."""
    opens, closes = beat - WINDOW, beat + WINDOW
    for start, end in runs:
        if start < opens - _EPS:
            if end >= opens - _EPS:
                return -WINDOW  # already in the pose when the window opened
            continue
        return start - beat if start <= closes + _EPS else None
    return None


def rate(offset: float | None) -> tuple[str, int]:
    if offset is not None:
        for limit, rating, points in TIERS:
            if abs(offset) <= limit + _EPS:
                return rating, points
    return MISS


def player_name(names: list[str], index: int) -> str | None:
    """The typed name for the index-th player from the left, if any."""
    if index < len(names) and names[index].strip():
        return names[index].strip()
    return None


def score_players(slots: list[dict], script: Script,
                  names: list[str]) -> list[PlayerScore]:
    """Score every tracked player (``yolo.extract_players`` slots, left -> right)."""
    players = []
    for i, slot in enumerate(slots):
        runs = {act: _runs(sorted(times)) for act, times in slot["actions"].items()}
        hits = []
        for step in script.steps:
            offset = step_offset(runs.get(step.move, []), step.at)
            rating, points = rate(offset)
            hits.append(MoveHit(step.move, step.at, offset, rating, points))
        name = player_name(names, i)
        players.append(PlayerScore(number=i + 1, name=name or f"Person {i + 1}",
                                   hits=hits, named=name is not None))
    return players


def pick_winner(players: list[PlayerScore]) -> int | None:
    """Index of the best total (ties: smallest summed |offset|); None if nobody scored."""
    scored = [i for i, p in enumerate(players) if p.total > 0]
    if not scored:
        return None
    return max(scored, key=lambda i: (players[i].total, -players[i].abs_error))
