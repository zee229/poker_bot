"""Tests for MCCFR — strategy validity and convergence on Leduc."""

import numpy as np
import pytest

from poker_bot.ai.mccfr import MCCFR
from poker_bot.ai.vanilla_cfr import VanillaCFR
from poker_bot.games.leduc import LeducPoker


class TestLeducVanillaCFR:
    @pytest.fixture
    def trained(self):
        leduc = LeducPoker()
        cfr = VanillaCFR(leduc)
        for _ in range(1_000):
            cfr.iterate()
        return cfr

    def test_info_set_count(self, trained):
        assert len(trained.info_sets) > 50

    def test_exploitability_decreases(self):
        leduc = LeducPoker()
        cfr = VanillaCFR(leduc)
        for _ in range(200):
            cfr.iterate()
        exploit_200 = cfr.compute_exploitability()

        for _ in range(800):
            cfr.iterate()
        exploit_1k = cfr.compute_exploitability()

        assert exploit_1k < exploit_200

    def test_strategies_valid(self, trained):
        for key in trained.info_sets:
            strategy = trained.info_sets[key].get_average_strategy()
            assert abs(np.sum(strategy) - 1.0) < 1e-6
            assert np.all(strategy >= -1e-6)


class TestLeducMCCFR:
    @pytest.fixture
    def trained(self):
        leduc = LeducPoker()
        cfr = MCCFR(leduc, seed=42)
        for _ in range(10_000):
            cfr.iterate()
        return cfr

    def test_info_sets_populated(self, trained):
        assert len(trained.info_sets) > 50

    def test_strategies_valid(self, trained):
        for key in trained.info_sets:
            strategy = trained.info_sets[key].get_average_strategy()
            assert abs(np.sum(strategy) - 1.0) < 1e-6
            assert np.all(strategy >= -1e-6)

    def test_no_crash_many_iterations(self):
        leduc = LeducPoker()
        cfr = MCCFR(leduc, seed=123)
        for _ in range(5000):
            cfr.iterate()
        assert cfr.iterations == 5000
