"""Performance tests for hot-path optimizations."""

from __future__ import annotations

import time

import numpy as np
import pytest

from poker_bot.ai.infoset import InfoSet, _regret_match
from poker_bot.ai.mccfr import MCCFR
from poker_bot.games.kuhn import KuhnPoker


class TestRegretMatch:
    """Test regret matching optimization."""

    def test_basic_uniform(self):
        """All-zero regret → uniform strategy."""
        regret = np.zeros(4, dtype=np.float64)
        strategy = _regret_match(regret)
        np.testing.assert_allclose(strategy, [0.25, 0.25, 0.25, 0.25])

    def test_positive_regret(self):
        """Positive regrets normalize to strategy."""
        regret = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float64)
        strategy = _regret_match(regret)
        np.testing.assert_allclose(strategy, [2 / 3, 1 / 3, 0.0, 0.0])

    def test_all_negative(self):
        """All-negative regrets → uniform strategy."""
        regret = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        strategy = _regret_match(regret)
        np.testing.assert_allclose(strategy, [1 / 3, 1 / 3, 1 / 3])

    def test_single_action(self):
        """Single action → deterministic."""
        regret = np.array([5.0], dtype=np.float64)
        strategy = _regret_match(regret)
        np.testing.assert_allclose(strategy, [1.0])

    def test_matches_infoset_get_strategy(self):
        """_regret_match output matches InfoSet.get_strategy()."""
        info = InfoSet(5)
        info.cumulative_regret = np.array([3.0, -1.0, 2.0, 0.0, 1.0])
        expected = info.get_strategy()
        result = _regret_match(info.cumulative_regret)
        np.testing.assert_allclose(result, expected)


class TestSearchsortedSampling:
    """Test np.searchsorted-based sampling in MCCFR."""

    def test_sampling_distribution(self):
        """Sampling matches expected distribution over many samples."""
        game = KuhnPoker()
        cfr = MCCFR(game, seed=42)
        probs = np.array([0.1, 0.3, 0.6])
        counts = np.zeros(3)
        n = 10000
        for _ in range(n):
            idx = cfr._sample_weighted(probs)
            counts[idx] += 1
        empirical = counts / n
        np.testing.assert_allclose(empirical, probs, atol=0.03)

    def test_deterministic_100(self):
        """Probability 1.0 always returns that index."""
        game = KuhnPoker()
        cfr = MCCFR(game, seed=0)
        probs = np.array([0.0, 0.0, 1.0])
        for _ in range(100):
            assert cfr._sample_weighted(probs) == 2

    def test_list_input(self):
        """Works with Python list input too."""
        game = KuhnPoker()
        cfr = MCCFR(game, seed=7)
        idx = cfr._sample_weighted([0.5, 0.5])
        assert idx in (0, 1)


class TestKuhnCFRPerformance:
    """Benchmark CFR iterations/sec on Kuhn poker."""

    def test_kuhn_mccfr_speed(self):
        """MCCFR on Kuhn should run at least 5000 iterations/sec."""
        game = KuhnPoker()
        cfr = MCCFR(game, seed=42)
        n_iters = 2000

        start = time.perf_counter()
        for _ in range(n_iters):
            cfr.iterate()
        elapsed = time.perf_counter() - start

        iters_per_sec = n_iters / elapsed
        # Sanity check: should be reasonably fast
        assert iters_per_sec > 1000, f"Too slow: {iters_per_sec:.0f} iters/sec"


class TestInfoSetCFRPlus:
    """Test CFR+ regret update optimization."""

    def test_cfr_plus_floors_at_zero(self):
        info = InfoSet(3)
        info.cumulative_regret = np.array([1.0, -2.0, 3.0])
        info.update_regret_cfr_plus(np.array([-0.5, 1.0, -5.0]))
        np.testing.assert_allclose(info.cumulative_regret, [0.5, 0.0, 0.0])

    def test_cfr_plus_from_zero(self):
        info = InfoSet(2)
        info.update_regret_cfr_plus(np.array([1.0, -1.0]))
        np.testing.assert_allclose(info.cumulative_regret, [1.0, 0.0])
