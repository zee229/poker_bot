"""Kuhn Poker — minimal poker game for CFR validation.

3 cards (J, Q, K), 2 players, 1 card each.
Actions: check (pass) or bet 1 chip.
12 information sets, known Nash equilibrium.

Nash equilibrium for Kuhn:
- Player 0 with J: bet ~1/3, check ~2/3 (bluff)
- Player 0 with Q: always check
- Player 0 with K: always bet
- Player 1 with J facing bet: always fold
- Player 1 with Q facing bet: call 1/3
- Player 1 with K facing bet: always call
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from poker_bot.ai.cfr_base import GameAdapter

CARDS = ["J", "Q", "K"]
ACTIONS = ["p", "b"]  # pass, bet


@dataclass(frozen=True)
class KuhnState:
    cards: tuple[str, ...]  # cards[0] = P0's card, cards[1] = P1's card
    history: str = ""

    @property
    def is_terminal(self) -> bool:
        h = self.history
        return h in ("pp", "pbp", "pbb", "bp", "bb")

    @property
    def current_player(self) -> int:
        return len(self.history) % 2

    def utility(self, player: int) -> float:
        """Compute utility for given player at terminal state."""
        h = self.history
        card_values = {"J": 0, "Q": 1, "K": 2}
        p0_val = card_values[self.cards[0]]
        p1_val = card_values[self.cards[1]]
        winner = 0 if p0_val > p1_val else 1

        if h == "pp":
            # Both pass, showdown for 1 chip
            return 1.0 if winner == player else -1.0
        elif h == "bp":
            # P0 bets, P1 folds — P0 wins 1
            return 1.0 if player == 0 else -1.0
        elif h == "pbp":
            # P0 passes, P1 bets, P0 folds — P1 wins 1
            return 1.0 if player == 1 else -1.0
        elif h == "bb":
            # Both bet, showdown for 2 chips
            return 2.0 if winner == player else -2.0
        elif h == "pbb":
            # P0 passes, P1 bets, P0 calls — showdown for 2 chips
            return 2.0 if winner == player else -2.0
        return 0.0


class KuhnPoker(GameAdapter):
    def initial_state(self) -> str:
        """Return sentinel for chance node."""
        return "DEAL"

    def is_terminal(self, state) -> bool:
        if state == "DEAL":
            return False
        return state.is_terminal

    def terminal_utility(self, state, player: int) -> float:
        return state.utility(player)

    def current_player(self, state) -> int:
        if state == "DEAL":
            return -1  # chance node
        return state.current_player

    def num_players(self) -> int:
        return 2

    def info_set_key(self, state, player: int) -> str:
        return f"{state.cards[player]}:{state.history}"

    def legal_actions(self, state) -> list[str]:
        return ACTIONS  # always pass or bet

    def apply_action(self, state, action) -> KuhnState:
        if state == "DEAL":
            # action is a tuple of cards
            return KuhnState(cards=action, history="")
        return KuhnState(cards=state.cards, history=state.history + action)

    def chance_outcomes(self, state) -> list[tuple]:
        """All possible card dealings with equal probability."""
        deals = list(permutations(CARDS, 2))
        prob = 1.0 / len(deals)
        return [(deal, prob) for deal in deals]
