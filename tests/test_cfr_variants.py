"""Tests for CFR+ and DCFR variants."""

import numpy as np
import pytest

from poker_bot.ai.cfr_base import CFRVariant
from poker_bot.ai.infoset import InfoSet
from poker_bot.ai.mccfr import MCCFR
from poker_bot.ai.vanilla_cfr import VanillaCFR
from poker_bot.games.kuhn import KuhnPoker


class TestCFRPlusRegretUpdate:
    def test_regrets_always_nonnegative(self):
        info = InfoSet(3)
        info.update_regret_cfr_plus(np.array([-5.0, 2.0, -1.0]))
        assert np.all(info.cumulative_regret >= 0)

    def test_positive_regrets_accumulate(self):
        info = InfoSet(2)
        info.update_regret_cfr_plus(np.array([3.0, -1.0]))
        info.update_regret_cfr_plus(np.array([2.0, -1.0]))
        assert info.cumulative_regret[0] == 5.0
        assert info.cumulative_regret[1] == 0.0

    def test_floor_at_zero_each_step(self):
        info = InfoSet(2)
        info.update_regret_cfr_plus(np.array([-10.0, 5.0]))
        # -10 floored to 0
        assert info.cumulative_regret[0] == 0.0
        info.update_regret_cfr_plus(np.array([-1.0, 1.0]))
        # 0 + (-1) = -1, floored to 0
        assert info.cumulative_regret[0] == 0.0


class TestDCFRRegretUpdate:
    def test_basic_dcfr_update(self):
        info = InfoSet(2)
        info.cumulative_regret = np.array([10.0, -5.0])
        info.update_regret_dcfr(np.array([1.0, 1.0]), t=10)
        # Positive regret 10 * discount_pos + 1, negative -5 * discount_neg + 1
        assert info.cumulative_regret[0] > 10.0  # discounted + added
        assert info.cumulative_regret[1] > -5.0   # less negative after discount + add

    def test_dcfr_discount_increases_with_t(self):
        """Later iterations discount less (keep more history)."""
        info1 = InfoSet(2)
        info1.cumulative_regret = np.array([10.0, -5.0])
        info1.update_regret_dcfr(np.array([0.0, 0.0]), t=2)

        info2 = InfoSet(2)
        info2.cumulative_regret = np.array([10.0, -5.0])
        info2.update_regret_dcfr(np.array([0.0, 0.0]), t=100)

        # At t=100 discount is closer to 1, so more is preserved
        assert info2.cumulative_regret[0] > info1.cumulative_regret[0]

    def test_dcfr_strategy_discount(self):
        info = InfoSet(2)
        info.cumulative_strategy = np.array([10.0, 10.0])
        strategy = info.update_strategy_dcfr(1.0, t=5)
        # Strategy should be discounted then updated
        assert np.all(info.cumulative_strategy > 0)
        assert abs(strategy.sum() - 1.0) < 1e-6


class TestCFRPlusKuhnConvergence:
    @pytest.fixture
    def trained_cfr_plus(self):
        kuhn = KuhnPoker()
        cfr = VanillaCFR(kuhn, variant=CFRVariant.CFR_PLUS)
        for _ in range(10_000):
            cfr.iterate()
        return cfr

    def test_exploitability(self, trained_cfr_plus):
        exploit = trained_cfr_plus.compute_exploitability()
        assert exploit < 0.01, f"CFR+ exploitability too high: {exploit}"

    def test_strategies_valid(self, trained_cfr_plus):
        for key in trained_cfr_plus.info_sets:
            strategy = trained_cfr_plus.info_sets[key].get_average_strategy()
            assert abs(np.sum(strategy) - 1.0) < 1e-6
            assert np.all(strategy >= -1e-6)

    def test_regrets_nonnegative(self, trained_cfr_plus):
        for key in trained_cfr_plus.info_sets:
            info = trained_cfr_plus.info_sets[key]
            assert np.all(info.cumulative_regret >= 0)


class TestDCFRKuhnConvergence:
    @pytest.fixture
    def trained_dcfr(self):
        kuhn = KuhnPoker()
        cfr = VanillaCFR(kuhn, variant=CFRVariant.DCFR)
        for _ in range(20_000):
            cfr.iterate()
        return cfr

    def test_exploitability(self, trained_dcfr):
        exploit = trained_dcfr.compute_exploitability()
        assert exploit < 0.01, f"DCFR exploitability too high: {exploit}"

    def test_strategies_valid(self, trained_dcfr):
        for key in trained_dcfr.info_sets:
            strategy = trained_dcfr.info_sets[key].get_average_strategy()
            assert abs(np.sum(strategy) - 1.0) < 1e-6
            assert np.all(strategy >= -1e-6)


class TestMCCFRWithCFRPlus:
    @pytest.fixture
    def trained_mccfr_plus(self):
        kuhn = KuhnPoker()
        cfr = MCCFR(kuhn, seed=42, variant=CFRVariant.CFR_PLUS)
        for _ in range(50_000):
            cfr.iterate()
        return cfr

    def test_converges(self, trained_mccfr_plus):
        exploit = trained_mccfr_plus.compute_exploitability()
        assert exploit < 0.05, f"MCCFR+CFR+ exploitability too high: {exploit}"

    def test_regrets_nonnegative(self, trained_mccfr_plus):
        for key in trained_mccfr_plus.info_sets:
            info = trained_mccfr_plus.info_sets[key]
            assert np.all(info.cumulative_regret >= 0)
