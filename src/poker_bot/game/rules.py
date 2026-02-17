"""NLHE rules and blinds configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlindStructure:
    small_blind: int
    big_blind: int
    ante: int = 0

    @property
    def min_buyin(self) -> int:
        return self.big_blind * 20

    @property
    def max_buyin(self) -> int:
        return self.big_blind * 100


DEFAULT_BLINDS = BlindStructure(small_blind=50, big_blind=100)

MIN_PLAYERS = 2
MAX_PLAYERS = 6
CARDS_PER_PLAYER = 2
BOARD_SIZES = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}
