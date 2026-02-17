"""Suit isomorphism for reducing card combinations."""

from __future__ import annotations

from poker_bot.game.card import Card, Rank, Suit


def canonical_hand(cards: list[Card]) -> tuple[Card, ...]:
    """Canonicalize suits so that equivalent hands map to the same key.

    Assigns suits in order of first appearance. E.g.:
    Ah Kh -> As Ks (hearts mapped to spades as first suit)
    Ad Kd -> As Ks (same mapping)
    """
    suit_map: dict[Suit, Suit] = {}
    next_suit = iter(Suit)
    result = []

    for card in cards:
        if card.suit not in suit_map:
            suit_map[card.suit] = next(next_suit)
        result.append(Card(rank=card.rank, suit=suit_map[card.suit]))

    return tuple(result)


def preflop_hand_key(card1: Card, card2: Card) -> str:
    """Get canonical preflop hand string.

    Returns format like 'AKs' (suited) or 'AKo' (offsuit) or 'AA' (pair).
    169 unique canonical preflop hands.
    """
    r1, r2 = card1.rank, card2.rank
    high, low = (r1, r2) if r1 >= r2 else (r2, r1)

    if r1 == r2:
        return f"{high.char}{low.char}"
    elif card1.suit == card2.suit:
        return f"{high.char}{low.char}s"
    else:
        return f"{high.char}{low.char}o"


def all_preflop_hands() -> list[str]:
    """Return all 169 canonical preflop hands."""
    hands = []
    ranks = list(reversed(Rank))  # A, K, Q, ..., 2

    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i == j:
                hands.append(f"{r1.char}{r2.char}")
            elif i < j:
                hands.append(f"{r1.char}{r2.char}s")
            else:
                hands.append(f"{r2.char}{r1.char}o")

    return hands
