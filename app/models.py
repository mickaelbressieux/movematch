from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Step:
    """One beat of a script: perform ``move`` (a pose-library id) at ``at`` seconds
    after the recording starts. ``lead`` (song scripts) is how long before the beat
    the dancer's own pose already matched: a player copying her exactly starts that
    early too, so scoring aims at ``at - lead`` while the avatar still lands on the
    beat."""
    move: str
    at: float
    lead: float = 0.0

    @property
    def target(self) -> float:
        """When a perfectly timed player's pose starts matching."""
        return self.at - self.lead


@dataclass(frozen=True)
class Script:
    """A choreography: timed moves loaded from ``scripts/<id>.json`` (see dance.py)."""
    id: str
    name: str
    steps: tuple[Step, ...]
    audio: str | None = None       # song file under static/, if the script has music
    duration: float | None = None  # length of the song, s
    source: str | None = None      # song project (songs_src/<id>) it was made from

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
    speed: float = 1.0          # playback speed (< 1: slow practice)
    audio_offset: float = 0.0   # s into the recording when song time 0 played

    def to_json(self) -> dict:
        """Everything but the script (stored by id): a background job's result."""
        return {"script": self.script.id, "players": [asdict(p) for p in self.players],
                "winner": self.winner, "num_people": self.num_people,
                "winner_image": self.winner_image, "timeline": self.timeline,
                "speed": self.speed, "audio_offset": self.audio_offset}

    @classmethod
    def from_json(cls, data: dict, script: Script) -> GameResult:
        players = [PlayerScore(number=p["number"], name=p["name"], named=p["named"],
                               hits=[MoveHit(**h) for h in p["hits"]])
                   for p in data["players"]]
        return cls(script=script, players=players, winner=data["winner"],
                   num_people=data["num_people"], winner_image=data["winner_image"],
                   timeline=[tuple(e) for e in data["timeline"]], speed=data["speed"],
                   audio_offset=data["audio_offset"])

    @property
    def has_winner(self) -> bool:
        return self.winner is not None

    @property
    def winner_player(self) -> PlayerScore | None:
        return self.players[self.winner] if self.has_winner else None
