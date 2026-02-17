"""Tests for CFR on Kuhn Poker — validates convergence to Nash equilibrium."""

import numpy as np
import pytest

from poker_bot.ai.vanilla_cfr import VanillaCFR
from poker_bot.ai.mccfr import MCCFR
from poker_bot.games.kuhn import KuhnPoker


@pytest.fixture
def trained_vanilla():
    kuhn = KuhnPoker()
    cfr = VanillaCFR(kuhn)
    for _ in range(100_000):
        cfr.iterate()
    return cfr


@pytest.fixture
def trained_mccfr():
    kuhn = KuhnPoker()
    cfr = MCCFR(kuhn, seed=42)
    for _ in range(100_000):
        cfr.iterate()
    return cfr


class TestKuhnVanillaCFR:
    def test_info_set_count(self, trained_vanilla):
        assert len(trained_vanilla.info_sets) == 12

    def test_exploitability(self, trained_vanilla):
        exploit = trained_vanilla.compute_exploitability()
        assert exploit < 0.01, f"Exploitability too high: {exploit}"

    def test_p0_king_bet_ratio(self, trained_vanilla):
        """P0 K:bet should be ~3x J:bet (Nash family: K bets with 3*alpha)."""
        k_info = trained_vanilla.info_sets["K:"]
        j_info = trained_vanilla.info_sets["J:"]
        k_bet = k_info.get_average_strategy()[1]
        j_bet = j_info.get_average_strategy()[1]
        if j_bet > 0.01:
            ratio = k_bet / j_bet
            assert abs(ratio - 3.0) < 0.5, f"K/J bet ratio should be ~3, got {ratio:.2f}"

    def test_p0_queen_mostly_checks(self, trained_vanilla):
        info = trained_vanilla.info_sets["Q:"]
        strategy = info.get_average_strategy()
        assert strategy[0] > 0.9, f"Q should mostly check, got {strategy}"

    def test_p0_jack_bluff_bounded(self, trained_vanilla):
        """P0 J bluff frequency should be in [0, 1/3]."""
        info = trained_vanilla.info_sets["J:"]
        bet_freq = info.get_average_strategy()[1]
        assert 0 <= bet_freq <= 0.4, f"J bet should be in [0, 1/3], got {bet_freq:.3f}"

    def test_p1_king_facing_bet_always_calls(self, trained_vanilla):
        info = trained_vanilla.info_sets["K:b"]
        strategy = info.get_average_strategy()
        assert strategy[1] > 0.95, f"K facing bet should always call, got {strategy}"

    def test_p1_jack_facing_bet_always_folds(self, trained_vanilla):
        info = trained_vanilla.info_sets["J:b"]
        strategy = info.get_average_strategy()
        assert strategy[0] > 0.95, f"J facing bet should always fold, got {strategy}"

    def test_p1_queen_facing_bet_calls_third(self, trained_vanilla):
        """P1 Q facing bet should call ~1/3 of the time."""
        info = trained_vanilla.info_sets["Q:b"]
        call_freq = info.get_average_strategy()[1]
        assert abs(call_freq - 1 / 3) < 0.05, f"Q:b call should be ~1/3, got {call_freq:.3f}"

    def test_strategies_sum_to_one(self, trained_vanilla):
        for key in trained_vanilla.info_sets:
            strategy = trained_vanilla.info_sets[key].get_average_strategy()
            assert abs(np.sum(strategy) - 1.0) < 1e-6


class TestKuhnMCCFR:
    def test_converges(self, trained_mccfr):
        exploit = trained_mccfr.compute_exploitability()
        assert exploit < 0.05, f"MCCFR exploitability too high: {exploit}"

    def test_strategies_valid(self, trained_mccfr):
        for key in trained_mccfr.info_sets:
            strategy = trained_mccfr.info_sets[key].get_average_strategy()
            assert abs(np.sum(strategy) - 1.0) < 1e-6
            assert np.all(strategy >= -1e-6)
