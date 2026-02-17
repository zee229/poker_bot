"""Tests for card abstraction determinism and pipeline."""

import tempfile
from pathlib import Path

from poker_bot.ai.abstraction.card_abstraction import CardAbstraction, _fnv1a
from poker_bot.game.card import Card


class TestFnv1a:
    def test_deterministic(self):
        assert _fnv1a("hello") == _fnv1a("hello")

    def test_different_inputs(self):
        assert _fnv1a("hello") != _fnv1a("world")

    def test_returns_int(self):
        result = _fnv1a("test")
        assert isinstance(result, int)


class TestGetBucketDeterminism:
    def test_same_input_same_bucket(self):
        abs1 = CardAbstraction()
        abs2 = CardAbstraction()
        hole = [Card.from_str("Ah"), Card.from_str("Kh")]
        board = [Card.from_str("Qh"), Card.from_str("Jh"), Card.from_str("Th")]
        b1 = abs1.get_bucket("flop", hole, board)
        b2 = abs2.get_bucket("flop", hole, board)
        assert b1 == b2

    def test_fallback_within_range(self):
        abs_ = CardAbstraction(num_flop_buckets=50)
        hole = [Card.from_str("2c"), Card.from_str("3d")]
        board = [Card.from_str("7h"), Card.from_str("8s"), Card.from_str("9c")]
        bucket = abs_.get_bucket("flop", hole, board)
        assert 0 <= bucket < 50

    def test_preflop_fallback_deterministic(self):
        abs_ = CardAbstraction()
        hole = [Card.from_str("As"), Card.from_str("Ks")]
        b1 = abs_.get_bucket("preflop", hole, [])
        b2 = abs_.get_bucket("preflop", hole, [])
        assert b1 == b2


class TestSaveLoadRoundTrip:
    def test_round_trip(self):
        abs_ = CardAbstraction(num_river_buckets=5)
        abs_.compute_river_buckets(num_samples=100)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "abs"
            abs_.save(path)

            abs2 = CardAbstraction(num_river_buckets=5)
            abs2.load(path)

            # Check same keys
            assert set(abs_._lut["river"].keys()) == set(abs2._lut["river"].keys())
            for k in abs_._lut["river"]:
                assert abs_._lut["river"][k] == abs2._lut["river"][k]


class TestComputeTurnUsesRiverLUT:
    def test_turn_buckets_use_river_lut(self):
        abs_ = CardAbstraction(num_river_buckets=5, num_turn_buckets=3)
        # Must compute river first
        abs_.compute_river_buckets(num_samples=100)
        assert len(abs_._lut.get("river", {})) > 0

        abs_.compute_turn_buckets(num_samples=50, num_rollouts=10)
        assert len(abs_._lut.get("turn", {})) > 0

        # Buckets should be within range
        for v in abs_._lut["turn"].values():
            assert 0 <= v < 3


class TestComputeAll:
    def test_compute_all_populates_all_streets(self):
        abs_ = CardAbstraction(
            num_preflop_buckets=4,
            num_river_buckets=5,
            num_turn_buckets=3,
        )
        abs_.compute_all(river_samples=100, turn_samples=50, turn_rollouts=10)
        assert "river" in abs_._lut
        assert "turn" in abs_._lut
        assert "preflop" in abs_._lut
