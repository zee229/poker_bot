"""Tests for parallel MCCFR training."""

import numpy as np
import pytest

from poker_bot.ai.cfr_base import CFRVariant
from poker_bot.ai.infoset import InfoSetStore
from poker_bot.ai.parallel_mccfr import ParallelMCCFR, merge_info_sets, _worker_run
from poker_bot.games.kuhn import KuhnPoker


class TestParallelMCCFRKuhn:
    def test_two_workers_converge(self):
        kuhn = KuhnPoker()
        parallel = ParallelMCCFR(kuhn, num_workers=2, base_seed=42)
        parallel.train(50_000)

        exploit = parallel.compute_exploitability()
        assert exploit < 0.05, f"Parallel MCCFR exploitability too high: {exploit}"

    def test_merged_info_set_count(self):
        kuhn = KuhnPoker()
        parallel = ParallelMCCFR(kuhn, num_workers=2, base_seed=42)
        parallel.train(10_000)

        # Kuhn has 12 info sets
        assert len(parallel.info_sets) == 12

    def test_strategies_valid(self):
        kuhn = KuhnPoker()
        parallel = ParallelMCCFR(kuhn, num_workers=2, base_seed=42)
        parallel.train(10_000)

        for key in parallel.info_sets:
            strategy = parallel.info_sets[key].get_average_strategy()
            assert abs(np.sum(strategy) - 1.0) < 1e-6
            assert np.all(strategy >= -1e-6)


class TestWorkerRun:
    def test_different_seeds_different_regrets(self):
        kuhn = KuhnPoker()
        r1 = _worker_run(kuhn, 1000, seed=1, variant=CFRVariant.VANILLA)
        r2 = _worker_run(kuhn, 1000, seed=999, variant=CFRVariant.VANILLA)

        # Both should produce 12 info sets
        assert len(r1) == 12
        assert len(r2) == 12

        # Regrets should differ (different random samples)
        some_key = next(iter(r1))
        assert not np.array_equal(r1[some_key][0], r2[some_key][0])


class TestMergeInfoSets:
    def test_merge_adds_regrets(self):
        store = InfoSetStore()
        results = [
            {"K:": (np.array([1.0, 2.0]), np.array([0.5, 0.5]), 2)},
            {"K:": (np.array([3.0, -1.0]), np.array([0.3, 0.7]), 2)},
        ]
        merge_info_sets(store, results)

        info = store["K:"]
        np.testing.assert_array_almost_equal(info.cumulative_regret, [4.0, 1.0])
        np.testing.assert_array_almost_equal(info.cumulative_strategy, [0.8, 1.2])

    def test_merge_creates_new_keys(self):
        store = InfoSetStore()
        results = [
            {"K:": (np.array([1.0, 2.0]), np.array([0.5, 0.5]), 2)},
            {"Q:": (np.array([3.0, -1.0]), np.array([0.3, 0.7]), 2)},
        ]
        merge_info_sets(store, results)
        assert "K:" in store
        assert "Q:" in store


class TestParallelWithCFRPlus:
    def test_cfr_plus_variant(self):
        kuhn = KuhnPoker()
        parallel = ParallelMCCFR(
            kuhn, num_workers=2, base_seed=42, variant=CFRVariant.CFR_PLUS,
        )
        parallel.train(20_000)

        exploit = parallel.compute_exploitability()
        assert exploit < 0.1, f"Parallel CFR+ exploitability too high: {exploit}"
