"""Tests for abstraction layer."""

from poker_bot.ai.abstraction.isomorphism import (
    all_preflop_hands,
    canonical_hand,
    preflop_hand_key,
)
from poker_bot.game.card import Card


class TestSuitIsomorphism:
    def test_canonical_same_suit(self):
        cards1 = [Card.from_str("Ah"), Card.from_str("Kh")]
        cards2 = [Card.from_str("Ad"), Card.from_str("Kd")]
        assert canonical_hand(cards1) == canonical_hand(cards2)

    def test_canonical_different_suit(self):
        cards1 = [Card.from_str("Ah"), Card.from_str("Kd")]
        cards2 = [Card.from_str("As"), Card.from_str("Kc")]
        assert canonical_hand(cards1) == canonical_hand(cards2)

    def test_canonical_preserves_order(self):
        cards = [Card.from_str("Ah"), Card.from_str("Kh")]
        canon = canonical_hand(cards)
        assert canon[0].rank.char == "A"
        assert canon[1].rank.char == "K"
        assert canon[0].suit == canon[1].suit


class TestPreflopHandKey:
    def test_pair(self):
        key = preflop_hand_key(Card.from_str("As"), Card.from_str("Ah"))
        assert key == "AA"

    def test_suited(self):
        key = preflop_hand_key(Card.from_str("As"), Card.from_str("Ks"))
        assert key == "AKs"

    def test_offsuit(self):
        key = preflop_hand_key(Card.from_str("As"), Card.from_str("Kh"))
        assert key == "AKo"

    def test_order_independent(self):
        key1 = preflop_hand_key(Card.from_str("As"), Card.from_str("Kh"))
        key2 = preflop_hand_key(Card.from_str("Kh"), Card.from_str("As"))
        assert key1 == key2


class TestAllPreflopHands:
    def test_169_hands(self):
        hands = all_preflop_hands()
        assert len(hands) == 169

    def test_includes_pairs(self):
        hands = all_preflop_hands()
        assert "AA" in hands
        assert "22" in hands

    def test_includes_suited(self):
        hands = all_preflop_hands()
        assert "AKs" in hands
        assert "76s" in hands

    def test_includes_offsuit(self):
        hands = all_preflop_hands()
        assert "AKo" in hands
        assert "72o" in hands
