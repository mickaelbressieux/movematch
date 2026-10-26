"""MoveMatch dance scripts and timing scores.

A script (``scripts/<id>.json``) is a list of timed steps::

    {"id": "warmup", "name": "Warm-up",
     "steps": [{"move": "clap", "at": 2.0}, {"move": "raise_one_arm", "at": 4.0}]}

Song scripts (made from a dance video by choreo.py) add ``"audio"`` (a path under
static/), ``"duration"``, ``"source"`` (the song project they came from) and a
per-step ``"lead"``: scoring aims ``lead`` seconds before the beat (see Step).

``move`` is a pose-library id (see poses.py) and ``at`` the beat time in seconds of
song time (for a script without music: from the start of the recording). While
dancing, each move's avatar slides in ``AVATAR_LEAD`` seconds of song time before
its beat.

Scoring works per player and per step. The frames where the player's pose matches
the step's template are merged into runs (a gap shorter than ``DEBOUNCE`` doesn't
end a run), and the first run that STARTS within the step's window gives the
player's offset. The window reaches half-way to each neighbouring step, capped at
``WINDOW``, so dense moves never share a window. A run already going when the window
opens counts as the earliest allowed start, unless an earlier step already took
that run (holding one pose doesn't score two beats). The offset maps to a tier;
tiers are in real seconds at every speed.

Clocks: the browser reports when song time 0 happened in the recording
(``audio_offset``: after a silent pre-roll, measured against the music) and the
playback ``speed`` (slow practice: 0.75, 0.5), so a step's moment in the clip is
``audio_offset + target / speed``.
"""

from __future__ import annotations

import json
from pathlib import Path

import poses
from models import MoveHit, PlayerScore, Script, Step

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"

AVATAR_LEAD = 2.0   # s an avatar takes to slide from the right edge to the hit line
WINDOW = 1.0        # s, most a move's window reaches before / after its beat
SPEEDS = (1.0, 0.75, 0.5)  # playback speeds offered (slow practice)
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
    steps = tuple(Step(str(s["move"]), float(s["at"]), float(s.get("lead", 0.0)))
                  for s in data["steps"])
    if not steps:
        raise ValueError(f"{path.name}: no steps")
    known = poses.load_library()
    for prev, step in zip((None, *steps), steps):
        if step.move not in known:
            raise ValueError(f"{path.name}: unknown move {step.move!r} "
                             f"(pose library: {', '.join(sorted(known))})")
        if prev and step.at <= prev.at:
            raise ValueError(f"{path.name}: step times must increase ({step.at})")
    return Script(id=str(data["id"]), name=str(data["name"]), steps=steps,
                  audio=data.get("audio"), duration=data.get("duration"),
                  source=data.get("source"))


_scripts: tuple[tuple, dict[str, Script]] = ((), {})


def load_scripts() -> dict[str, Script]:
    """All scripts by id, shortest first. Raises on an invalid file or a duplicate
    id. Reloaded when a file changes, so a script saved from the review page (by
    any server worker) shows up without a restart."""
    global _scripts
    paths = sorted(SCRIPT_DIR.glob("*.json"))
    stamp = tuple((p.name, p.stat().st_mtime_ns) for p in paths)
    if stamp == _scripts[0]:
        return _scripts[1]
    scripts: dict[str, Script] = {}
    for path in paths:
        script = _parse(path)
        if script.id in scripts:
            raise ValueError(f"{path.name}: duplicate script id {script.id!r}")
        scripts[script.id] = script
    _scripts = (stamp, dict(sorted(scripts.items(), key=lambda kv: len(kv[1].steps))))
    return _scripts[1]


def get_script(script_id: str | None) -> Script | None:
    return load_scripts().get(script_id or "")


def end_time(script: Script) -> float:
    """When recording stops for this script."""
    return script.steps[-1].at + END_GRACE


# -- Move detection -----------------------------------------------------------

TIMELINE_GAP = 0.6  # s without the pose that splits two timeline intervals


def detect_moves(slots: list[dict], script: Script) -> None:
    """Fill each slot's ``actions`` with ``{move: [times the pose matched]}`` for
    the moves this script uses."""
    lib = poses.load_library()
    for slot in slots:
        slot["actions"] = {m: poses.match_times(slot["poses"], lib[m])
                           for m in set(script.moves)}


def build_timeline(slots: list[dict]) -> list[tuple]:
    """Each player's detected moves as readable ``(start, end, text)`` intervals."""
    events = []
    for rank, s in enumerate(slots, start=1):
        for move, times in s["actions"].items():
            if not times:
                continue
            text = f"Person {rank}: {poses.label(move)}"
            start = prev = times[0]
            for t in times[1:]:
                if t - prev > TIMELINE_GAP:
                    events.append((start, prev, text))
                    start = t
                prev = t
            events.append((start, prev, text))
    return sorted(events, key=lambda e: e[0])


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


def step_windows(beats: list[float]) -> list[tuple[float, float]]:
    """(before, after) reach of each beat's window: half the gap to the neighbour on
    that side, at most WINDOW."""
    out = []
    for i, b in enumerate(beats):
        lo = min(WINDOW, (b - beats[i - 1]) / 2) if i else WINDOW
        hi = min(WINDOW, (beats[i + 1] - b) / 2) if i + 1 < len(beats) else WINDOW
        out.append((lo, hi))
    return out


def step_offset(runs: list[tuple[float, float]], beat: float, lo: float = WINDOW,
                hi: float = WINDOW) -> tuple[float, tuple[float, float]] | None:
    """(signed start offset vs ``beat``, the run used) from a move's runs, looking
    ``lo`` before and ``hi`` after the beat; None if missed."""
    opens, closes = beat - lo, beat + hi
    for run in runs:
        start, end = run
        if start < opens - _EPS:
            if end >= opens - _EPS:
                return -lo, run  # already in the pose when the window opened
            continue
        return (start - beat, run) if start <= closes + _EPS else None
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


def clip_times(script: Script, audio_offset: float = 0.0, speed: float = 1.0) -> list[float]:
    """When each step's target falls in the recording (see the module docstring)."""
    return [audio_offset + step.target / speed for step in script.steps]


def score_players(slots: list[dict], script: Script, names: list[str],
                  audio_offset: float = 0.0, speed: float = 1.0) -> list[PlayerScore]:
    """Score every tracked player (``yolo.extract_players`` slots, left -> right)."""
    beats = clip_times(script, audio_offset, speed)
    windows = step_windows(beats)
    players = []
    for i, slot in enumerate(slots):
        runs = {act: _runs(sorted(times)) for act, times in slot["actions"].items()}
        used: set[tuple[str, tuple]] = set()  # (move, run) already credited
        hits = []
        for step, beat, (lo, hi) in zip(script.steps, beats, windows):
            free = [r for r in runs.get(step.move, []) if (step.move, r) not in used]
            found = step_offset(free, beat, lo, hi)
            offset = None
            if found:
                offset, run = found
                used.add((step.move, run))
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
