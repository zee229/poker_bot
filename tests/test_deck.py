"""Tests for Deck."""

from poker_bot.game.deck import Deck


class TestDeck:
    def test_52_cards(self):
        d = Deck(seed=42)
        d.shuffle()
        assert d.remaining == 52

    def test_deal(self):
        d = Deck(seed=42)
        d.shuffle()
        cards = d.deal(5)
        assert len(cards) == 5
        assert d.remaining == 47

    def test_deal_one(self):
        d = Deck(seed=42)
        d.shuffle()
        c = d.deal_one()
        assert d.remaining == 51

    def test_seed_determinism(self):
        d1 = Deck(seed=42)
        d1.shuffle()
        cards1 = d1.deal(10)

        d2 = Deck(seed=42)
        d2.shuffle()
        cards2 = d2.deal(10)

        assert cards1 == cards2

    def test_different_seeds(self):
        d1 = Deck(seed=42)
        d1.shuffle()
        cards1 = d1.deal(10)

        d2 = Deck(seed=99)
        d2.shuffle()
        cards2 = d2.deal(10)

        assert cards1 != cards2

    def test_deal_too_many(self):
        d = Deck(seed=42)
        d.shuffle()
        d.deal(52)
        try:
            d.deal(1)
            assert False, "Should have raised"
        except RuntimeError:
            pass

    def test_shuffle_resets_index(self):
        d = Deck(seed=42)
        d.shuffle()
        d.deal(10)
        d.shuffle()
        assert d.remaining == 52

    def test_no_duplicates(self):
        d = Deck(seed=42)
        d.shuffle()
        cards = d.deal(52)
        assert len(set(cards)) == 52
