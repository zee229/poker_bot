"""Leduc Hold'em — small poker game for CFR validation.

6 cards: J♠ J♦ Q♠ Q♦ K♠ K♦
2 players, 1 hole card each, 1 community card.
2 betting rounds: pre-community and post-community.
Bet sizes: 2 chips first round, 4 chips second round.
Max 2 raises per round.
Pair > high card. Higher pair/card wins ties.
~936 info sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

from poker_bot.ai.cfr_base import GameAdapter

CARDS = ["Js", "Jd", "Qs", "Qd", "Ks", "Kd"]
ACTIONS = ["f", "c", "r"]  # fold, call/check, raise
RANK_VALUES = {"J": 0, "Q": 1, "K": 2}


@dataclass(frozen=True)
class LeducState:
    hole_cards: tuple[str, str]  # (P0 card, P1 card)
    community: str = ""
    history: tuple[str, ...] = ()
    round_num: int = 0
    bets: tuple[int, int] = (1, 1)  # ante
    raises_this_round: int = 0
    folded: int = -1  # player who folded, -1 if none
    is_deal_community: bool = False  # true if we need to deal community card

    @property
    def is_terminal(self) -> bool:
        if self.folded >= 0:
            return True
        if self.round_num >= 2:
            return True
        return False

    @property
    def current_player(self) -> int:
        round_actions = self._round_actions()
        return len(round_actions) % 2

    def _round_actions(self) -> list[str]:
        """Actions in current round."""
        # Find start of current round
        start = 0
        round_count = 0
        for i, a in enumerate(self.history):
            if a == "|":
                round_count += 1
                start = i + 1
                if round_count >= self.round_num:
                    break
        return list(self.history[start:])

    def utility(self, player: int) -> float:
        if self.folded >= 0:
            # Player who didn't fold wins the pot
            winner = 1 - self.folded
            pot = sum(self.bets)
            return float(pot - self.bets[player]) if player == winner else -float(self.bets[player])

        # Showdown
        p0_rank = RANK_VALUES[self.hole_cards[0][0]]
        p1_rank = RANK_VALUES[self.hole_cards[1][0]]
        comm_rank = RANK_VALUES[self.community[0]] if self.community else -1

        # Check for pairs
        p0_pair = p0_rank == comm_rank
        p1_pair = p1_rank == comm_rank

        if p0_pair and not p1_pair:
            winner = 0
        elif p1_pair and not p0_pair:
            winner = 1
        elif p0_rank > p1_rank:
            winner = 0
        elif p1_rank > p0_rank:
            winner = 1
        else:
            # True tie
            pot = sum(self.bets)
            return 0.0

        pot = sum(self.bets)
        return float(pot - self.bets[player]) if player == winner else -float(self.bets[player])


class LeducPoker(GameAdapter):
    MAX_RAISES = 2
    BET_SIZES = [2, 4]  # round 0 = 2, round 1 = 4

    def initial_state(self):
        return "DEAL"

    def is_terminal(self, state) -> bool:
        if isinstance(state, str):
            return False
        return state.is_terminal

    def terminal_utility(self, state, player: int) -> float:
        return state.utility(player)

    def current_player(self, state) -> int:
        if isinstance(state, str):
            return -1
        if state.is_deal_community:
            return -1
        return state.current_player

    def num_players(self) -> int:
        return 2

    def info_set_key(self, state, player: int) -> str:
        card = state.hole_cards[player]
        # Canonicalize: only rank matters for the hole card, but community suit matters
        card_rank = card[0]
        comm = state.community[0] if state.community else ""
        # Check if pair
        pair = "P" if (comm and comm == card_rank) else ""
        history_str = "".join(state.history)
        return f"{card_rank}{pair}|{comm}|{history_str}"

    def legal_actions(self, state) -> list[str]:
        if isinstance(state, str):
            return []

        round_actions = state._round_actions()
        actions = []

        if not round_actions:
            # First to act: check or raise
            return ["c", "r"]

        last = round_actions[-1]
        if last == "r":
            # Facing raise: fold, call, or re-raise (if allowed)
            actions = ["f", "c"]
            if state.raises_this_round < self.MAX_RAISES:
                actions.append("r")
        else:
            # After check: check or raise
            actions = ["c", "r"]

        return actions

    def apply_action(self, state, action):
        if isinstance(state, str) and state == "DEAL":
            # action = (hole0, hole1)
            return LeducState(hole_cards=action)

        if state.is_deal_community:
            # action = community card
            return LeducState(
                hole_cards=state.hole_cards,
                community=action,
                history=state.history + ("|",),
                round_num=1,
                bets=state.bets,
                raises_this_round=0,
            )

        bet_size = self.BET_SIZES[state.round_num]
        new_history = state.history + (action,)

        if action == "f":
            return LeducState(
                hole_cards=state.hole_cards,
                community=state.community,
                history=new_history,
                round_num=state.round_num,
                bets=state.bets,
                raises_this_round=state.raises_this_round,
                folded=state.current_player,
            )

        if action == "r":
            # Raise: add bet_size to current player's bet
            bets = list(state.bets)
            cp = state.current_player
            bets[cp] += bet_size
            new_raises = state.raises_this_round + 1

            return LeducState(
                hole_cards=state.hole_cards,
                community=state.community,
                history=new_history,
                round_num=state.round_num,
                bets=tuple(bets),
                raises_this_round=new_raises,
            )

        if action == "c":
            # Check or call
            round_actions = state._round_actions()

            if round_actions and round_actions[-1] == "r":
                # Calling a raise: match opponent's bet
                bets = list(state.bets)
                cp = state.current_player
                opp = 1 - cp
                bets[cp] = bets[opp]

                # Round ends
                if state.round_num == 0 and not state.community:
                    # Need to deal community card
                    return LeducState(
                        hole_cards=state.hole_cards,
                        community=state.community,
                        history=new_history,
                        round_num=state.round_num,
                        bets=tuple(bets),
                        raises_this_round=0,
                        is_deal_community=True,
                    )
                else:
                    # Showdown
                    return LeducState(
                        hole_cards=state.hole_cards,
                        community=state.community,
                        history=new_history,
                        round_num=state.round_num + 1,
                        bets=tuple(bets),
                    )
            else:
                # Check
                if len(round_actions) >= 1:
                    # Both checked, round ends
                    if state.round_num == 0 and not state.community:
                        return LeducState(
                            hole_cards=state.hole_cards,
                            community=state.community,
                            history=new_history,
                            round_num=state.round_num,
                            bets=state.bets,
                            raises_this_round=0,
                            is_deal_community=True,
                        )
                    else:
                        return LeducState(
                            hole_cards=state.hole_cards,
                            community=state.community,
                            history=new_history,
                            round_num=state.round_num + 1,
                            bets=state.bets,
                        )
                else:
                    return LeducState(
                        hole_cards=state.hole_cards,
                        community=state.community,
                        history=new_history,
                        round_num=state.round_num,
                        bets=state.bets,
                        raises_this_round=state.raises_this_round,
                    )

    def chance_outcomes(self, state) -> list[tuple]:
        if isinstance(state, str) and state == "DEAL":
            deals = list(permutations(CARDS, 2))
            prob = 1.0 / len(deals)
            return [(deal, prob) for deal in deals]

        if isinstance(state, LeducState) and state.is_deal_community:
            # Deal community card from remaining cards
            used = set(state.hole_cards)
            remaining = [c for c in CARDS if c not in used]
            prob = 1.0 / len(remaining)
            return [(c, prob) for c in remaining]

        return []
