"""Hand evaluation using eval7 and Monte Carlo equity estimation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum

import eval7

from poker_bot.game.card import Card


class HandRank(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


_EVAL7_TYPE_TO_RANK = {
    0: HandRank.HIGH_CARD,
    1: HandRank.PAIR,
    2: HandRank.TWO_PAIR,
    3: HandRank.THREE_OF_A_KIND,
    4: HandRank.STRAIGHT,
    5: HandRank.FLUSH,
    6: HandRank.FULL_HOUSE,
    7: HandRank.FOUR_OF_A_KIND,
    8: HandRank.STRAIGHT_FLUSH,
}


@dataclass(frozen=True, slots=True)
class HandResult:
    score: int
    hand_rank: HandRank

    def __lt__(self, other: HandResult) -> bool:
        return self.score < other.score

    def __gt__(self, other: HandResult) -> bool:
        return self.score > other.score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        return self.score == other.score


def evaluate_hand(cards: list[Card]) -> HandResult:
    """Evaluate a 5-7 card hand using eval7."""
    if not (5 <= len(cards) <= 7):
        raise ValueError(f"Need 5-7 cards, got {len(cards)}")
    e7_cards = [c.eval7_card for c in cards]
    score = eval7.evaluate(e7_cards)
    hand_type = score >> 24
    hand_rank = _EVAL7_TYPE_TO_RANK.get(hand_type, HandRank.HIGH_CARD)
    return HandResult(score=score, hand_rank=hand_rank)


def best_hand(hole_cards: list[Card], board: list[Card]) -> HandResult:
    """Find best 5-card hand from hole cards + board (up to 7 cards)."""
    all_cards = hole_cards + board
    if len(all_cards) < 5:
        raise ValueError(f"Need at least 5 cards total, got {len(all_cards)}")
    return evaluate_hand(all_cards[:7])


def monte_carlo_equity(
    hole_cards: list[Card],
    board: list[Card],
    num_opponents: int = 1,
    num_simulations: int = 1000,
    seed: int | None = None,
) -> float:
    """Estimate equity via Monte Carlo simulation."""
    rng = random.Random(seed)
    used = set(c.int_value for c in hole_cards + board)
    remaining = [Card.from_str(str(c)) for c in _all_cards() if c.int_value not in used]

    board_needed = 5 - len(board)
    cards_needed = board_needed + 2 * num_opponents
    if len(remaining) < cards_needed:
        raise ValueError("Not enough remaining cards for simulation")

    wins = 0.0
    for _ in range(num_simulations):
        rng.shuffle(remaining)
        idx = 0
        sim_board = board + remaining[idx : idx + board_needed]
        idx += board_needed

        my_hand = evaluate_hand(hole_cards + sim_board)

        lost = False
        tied = False
        for _ in range(num_opponents):
            opp_hole = remaining[idx : idx + 2]
            idx += 2
            opp_hand = evaluate_hand(opp_hole + sim_board)
            if opp_hand > my_hand:
                lost = True
                break
            elif opp_hand == my_hand:
                tied = True

        if not lost:
            wins += 0.5 if tied else 1.0

    return wins / num_simulations


def _all_cards() -> list[Card]:
    from poker_bot.game.card import Rank, Suit
    return [Card(rank=r, suit=s) for r in Rank for s in Suit]
