"""Tests for hand evaluation."""

from poker_bot.game.card import Card
from poker_bot.game.hand_eval import (
    HandRank,
    best_hand,
    evaluate_hand,
    monte_carlo_equity,
)


def cards(s: str) -> list[Card]:
    """Parse space-separated card string."""
    return [Card.from_str(c) for c in s.split()]


class TestEvaluateHand:
    def test_royal_flush(self):
        result = evaluate_hand(cards("As Ks Qs Js Ts"))
        assert result.hand_rank == HandRank.STRAIGHT_FLUSH

    def test_straight_flush(self):
        result = evaluate_hand(cards("9h 8h 7h 6h 5h"))
        assert result.hand_rank == HandRank.STRAIGHT_FLUSH

    def test_four_of_a_kind(self):
        result = evaluate_hand(cards("Ks Kh Kd Kc 2s"))
        assert result.hand_rank == HandRank.FOUR_OF_A_KIND

    def test_full_house(self):
        result = evaluate_hand(cards("As Ah Ad Ks Kh"))
        assert result.hand_rank == HandRank.FULL_HOUSE

    def test_flush(self):
        result = evaluate_hand(cards("As Ks Qs Js 9s"))
        assert result.hand_rank == HandRank.FLUSH

    def test_straight(self):
        result = evaluate_hand(cards("As Kh Qd Js Tc"))
        assert result.hand_rank == HandRank.STRAIGHT

    def test_three_of_a_kind(self):
        result = evaluate_hand(cards("As Ah Ad Ks Qh"))
        assert result.hand_rank == HandRank.THREE_OF_A_KIND

    def test_two_pair(self):
        result = evaluate_hand(cards("As Ah Ks Kh Qd"))
        assert result.hand_rank == HandRank.TWO_PAIR

    def test_pair(self):
        result = evaluate_hand(cards("As Ah Ks Qh Jd"))
        assert result.hand_rank == HandRank.PAIR

    def test_high_card(self):
        result = evaluate_hand(cards("As Kh Qd Js 9c"))
        assert result.hand_rank == HandRank.HIGH_CARD

    def test_7_card_eval(self):
        result = evaluate_hand(cards("As Ks Qs Js Ts 2h 3d"))
        assert result.hand_rank == HandRank.STRAIGHT_FLUSH

    def test_comparison(self):
        rf = evaluate_hand(cards("As Ks Qs Js Ts"))
        flush = evaluate_hand(cards("As Ks Qs Js 9s"))
        assert rf > flush


class TestBestHand:
    def test_finds_best(self):
        hole = cards("As Ks")
        board = cards("Qs Js Ts 2h 3d")
        result = best_hand(hole, board)
        assert result.hand_rank == HandRank.STRAIGHT_FLUSH


class TestMonteCarloEquity:
    def test_aces_high_equity(self):
        eq = monte_carlo_equity(
            cards("As Ah"), [], num_opponents=1, num_simulations=2000, seed=42
        )
        assert eq > 0.80  # AA ~85% vs 1 opponent

    def test_low_cards_lower_equity(self):
        eq_high = monte_carlo_equity(
            cards("As Ah"), [], num_opponents=1, num_simulations=2000, seed=42
        )
        eq_low = monte_carlo_equity(
            cards("2s 7h"), [], num_opponents=1, num_simulations=2000, seed=42
        )
        assert eq_high > eq_low

    def test_with_board(self):
        eq = monte_carlo_equity(
            cards("As Ks"),
            cards("Qs Js Ts"),
            num_opponents=1,
            num_simulations=1000,
            seed=42,
        )
        assert eq > 0.95  # Royal flush draw completed
