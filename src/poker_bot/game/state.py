"""Game state and street representation for poker."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from poker_bot.game.actions import Action
from poker_bot.game.card import Card
from poker_bot.game.player import Player, PlayerStatus
from poker_bot.game.rules import BlindStructure


class Street(enum.IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


@dataclass
class Pot:
    amount: int = 0
    eligible: list[int] = field(default_factory=list)  # seat indices


@dataclass
class GameState:
    players: list[Player]
    blinds: BlindStructure
    board: list[Card] = field(default_factory=list)
    street: Street = Street.PREFLOP
    pots: list[Pot] = field(default_factory=list)
    current_bet: int = 0
    min_raise: int = 0
    action_to: int = 0  # seat index of player to act
    dealer: int = 0
    action_history: list[list[tuple[int, Action]]] = field(default_factory=list)
    hand_over: bool = False
    winners: list[tuple[int, int]] | None = None  # (seat, amount) pairs
    last_raiser: int = -1
    num_actions_this_street: int = 0

    @property
    def main_pot(self) -> int:
        return sum(p.amount for p in self.pots)

    @property
    def active_players(self) -> list[Player]:
        return [p for p in self.players if p.is_active]

    @property
    def players_in_hand(self) -> list[Player]:
        return [p for p in self.players if p.is_in_hand]

    @property
    def num_active(self) -> int:
        return sum(1 for p in self.players if p.is_active)

    @property
    def num_in_hand(self) -> int:
        return sum(1 for p in self.players if p.is_in_hand)

    def street_actions(self, street: Street | None = None) -> list[tuple[int, Action]]:
        s = street if street is not None else self.street
        idx = int(s)
        if idx < len(self.action_history):
            return self.action_history[idx]
        return []

    def clone(self) -> GameState:
        """Fast shallow clone — avoids copy.deepcopy overhead."""
        return GameState(
            players=[p.clone() for p in self.players],
            blinds=self.blinds,
            board=list(self.board),
            street=self.street,
            pots=[Pot(amount=p.amount, eligible=list(p.eligible)) for p in self.pots],
            current_bet=self.current_bet,
            min_raise=self.min_raise,
            action_to=self.action_to,
            dealer=self.dealer,
            action_history=[list(s) for s in self.action_history],
            hand_over=self.hand_over,
            winners=list(self.winners) if self.winners else None,
            last_raiser=self.last_raiser,
            num_actions_this_street=self.num_actions_this_street,
        )
