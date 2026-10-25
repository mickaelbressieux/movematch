from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step:
    """One beat of a script: perform ``move`` (a YOLO action id) at ``at`` seconds
    after the recording starts."""
    move: str
    at: float


@dataclass(frozen=True)
class Script:
    """A choreography: timed moves loaded from ``scripts/<id>.json`` (see dance.py)."""
    id: str
    name: str
    steps: tuple[Step, ...]

    @property
    def moves(self) -> list[str]:
        return [s.move for s in self.steps]


@dataclass
class MoveHit:
    """How one player did on one step. ``offset`` is the signed delay between the
    beat and the moment they started the move (negative = early); None = missed."""
    move: str
    at: float
    offset: float | None
    rating: str
    points: int


@dataclass
class PlayerScore:
    number: int  # 1-based, left -> right
    name: str
    hits: list[MoveHit]
    named: bool = False  # name typed by the players (vs the "Person N" fallback)

    @property
    def total(self) -> int:
        return sum(h.points for h in self.hits)

    @property
    def abs_error(self) -> float:
        """Summed |offset| of the moves they hit (tie-breaker: lower is better)."""
        return sum(abs(h.offset) for h in self.hits if h.offset is not None)


@dataclass
class GameResult:
    """Outcome of one dance: every detected player's per-move timing score."""
    script: Script
    players: list[PlayerScore] = field(default_factory=list)
    winner: int | None = None  # index into ``players``
    num_people: int = 0
    winner_image: str | None = None
    timeline: list[tuple] = field(default_factory=list)  # [(start, end, text)]

    @property
    def has_winner(self) -> bool:
        return self.winner is not None

    @property
    def winner_player(self) -> PlayerScore | None:
        return self.players[self.winner] if self.has_winner else None
