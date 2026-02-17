"""Player state for poker game."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from poker_bot.game.card import Card


class PlayerStatus(enum.Enum):
    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all_in"
    OUT = "out"  # busted / sitting out


@dataclass
class Player:
    name: str
    seat: int
    stack: int
    hole_cards: list[Card] = field(default_factory=list)
    status: PlayerStatus = PlayerStatus.ACTIVE
    bet_this_street: int = 0
    bet_this_hand: int = 0
    is_human: bool = False

    @property
    def is_active(self) -> bool:
        return self.status == PlayerStatus.ACTIVE

    @property
    def is_in_hand(self) -> bool:
        return self.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)

    def reset_for_hand(self) -> None:
        self.hole_cards = []
        self.bet_this_street = 0
        self.bet_this_hand = 0
        if self.stack > 0:
            self.status = PlayerStatus.ACTIVE
        else:
            self.status = PlayerStatus.OUT

    def reset_street_bet(self) -> None:
        self.bet_this_street = 0

    def clone(self) -> Player:
        """Fast shallow clone — avoids copy.deepcopy overhead."""
        return Player(
            name=self.name, seat=self.seat, stack=self.stack,
            hole_cards=list(self.hole_cards), status=self.status,
            bet_this_street=self.bet_this_street,
            bet_this_hand=self.bet_this_hand, is_human=self.is_human,
        )
