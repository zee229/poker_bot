"""Tests for Card, Rank, Suit."""

import pytest
from poker_bot.game.card import Card, Rank, Suit, make_deck


class TestCard:
    def test_from_str(self):
        c = Card.from_str("As")
        assert c.rank == Rank.ACE
        assert c.suit == Suit.SPADES
        assert str(c) == "As"

    def test_from_str_ten(self):
        c = Card.from_str("Td")
        assert c.rank == Rank.TEN
        assert c.suit == Suit.DIAMONDS

    def test_from_str_two(self):
        c = Card.from_str("2c")
        assert c.rank == Rank.TWO
        assert c.suit == Suit.CLUBS

    def test_from_str_invalid(self):
        with pytest.raises(ValueError):
            Card.from_str("Xx")
        with pytest.raises(ValueError):
            Card.from_str("A")
        with pytest.raises(ValueError):
            Card.from_str("Asz")

    def test_pretty(self):
        c = Card.from_str("Kh")
        assert c.pretty() == "K♥"

    def test_eval7_conversion(self):
        c = Card.from_str("As")
        e7 = c.eval7_card
        assert str(e7) == "As"

    def test_int_value_unique(self):
        deck = make_deck()
        values = [c.int_value for c in deck]
        assert len(set(values)) == 52

    def test_frozen(self):
        c = Card.from_str("As")
        with pytest.raises(AttributeError):
            c.rank = Rank.KING


class TestMakeDeck:
    def test_52_cards(self):
        deck = make_deck()
        assert len(deck) == 52

    def test_unique(self):
        deck = make_deck()
        assert len(set(deck)) == 52

    def test_all_ranks(self):
        deck = make_deck()
        ranks = set(c.rank for c in deck)
        assert ranks == set(Rank)

    def test_all_suits(self):
        deck = make_deck()
        suits = set(c.suit for c in deck)
        assert suits == set(Suit)
