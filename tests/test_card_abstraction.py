"""Tests for card abstraction determinism and pipeline."""

import tempfile
from pathlib import Path

import numpy as np

from poker_bot.ai.abstraction.card_abstraction import (
    CardAbstraction,
    _emd_distance_matrix,
    _fnv1a,
)
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


class TestComputeFlopBuckets:
    def test_flop_lut_populated(self):
        abs_ = CardAbstraction(num_river_buckets=5, num_turn_buckets=3, num_flop_buckets=3)
        abs_.compute_river_buckets(num_samples=100)
        abs_.compute_turn_buckets(num_samples=50, num_rollouts=10)
        abs_.compute_flop_buckets(num_samples=50, num_rollouts=10)
        assert len(abs_._lut.get("flop", {})) > 0

    def test_flop_buckets_within_range(self):
        abs_ = CardAbstraction(num_river_buckets=5, num_turn_buckets=3, num_flop_buckets=4)
        abs_.compute_river_buckets(num_samples=100)
        abs_.compute_turn_buckets(num_samples=50, num_rollouts=10)
        abs_.compute_flop_buckets(num_samples=50, num_rollouts=10)
        for v in abs_._lut["flop"].values():
            assert 0 <= v < 4

    def test_get_bucket_returns_lut_value(self):
        abs_ = CardAbstraction(num_river_buckets=5, num_turn_buckets=3, num_flop_buckets=4)
        abs_.compute_river_buckets(num_samples=100)
        abs_.compute_turn_buckets(num_samples=50, num_rollouts=10)
        abs_.compute_flop_buckets(num_samples=50, num_rollouts=10)
        # Pick a key from the LUT and verify get_bucket returns it
        if abs_._lut["flop"]:
            key = next(iter(abs_._lut["flop"]))
            expected = abs_._lut["flop"][key]
            # Parse key back to cards
            hole_str, board_str = key.split("|")
            hole = [Card.from_str(hole_str[i:i+2]) for i in range(0, len(hole_str), 2)]
            board = [Card.from_str(board_str[i:i+2]) for i in range(0, len(board_str), 2)]
            assert abs_.get_bucket("flop", hole, board) == expected


class TestComputeAll:
    def test_compute_all_populates_all_streets(self):
        abs_ = CardAbstraction(
            num_preflop_buckets=4,
            num_river_buckets=5,
            num_turn_buckets=3,
            num_flop_buckets=3,
        )
        abs_.compute_all(
            river_samples=100, turn_samples=50, turn_rollouts=10,
            flop_samples=50, flop_rollouts=10,
        )
        assert "river" in abs_._lut
        assert "turn" in abs_._lut
        assert "flop" in abs_._lut
        assert "preflop" in abs_._lut


class TestEMDDistance:
    def test_identical_histograms_zero(self):
        data = np.array([[0.25, 0.25, 0.25, 0.25]])
        centroids = np.array([[0.25, 0.25, 0.25, 0.25]])
        dists = _emd_distance_matrix(data, centroids)
        assert abs(dists[0, 0]) < 1e-10

    def test_opposite_histograms(self):
        data = np.array([[1.0, 0.0, 0.0]])
        centroids = np.array([[0.0, 0.0, 1.0]])
        dists = _emd_distance_matrix(data, centroids)
        assert abs(dists[0, 0] - 2.0) < 1e-10

    def test_symmetric(self):
        a = np.array([[0.5, 0.3, 0.2]])
        b = np.array([[0.1, 0.4, 0.5]])
        d1 = _emd_distance_matrix(a, b)
        d2 = _emd_distance_matrix(b, a)
        assert abs(d1[0, 0] - d2[0, 0]) < 1e-10

    def test_batch_shape(self):
        data = np.random.dirichlet([1, 1, 1, 1], size=10)
        centroids = np.random.dirichlet([1, 1, 1, 1], size=3)
        dists = _emd_distance_matrix(data, centroids)
        assert dists.shape == (10, 3)

    def test_kmeans_with_emd_valid_labels(self):
        abs_ = CardAbstraction()
        data = np.random.dirichlet([1, 1, 1, 1, 1], size=50)
        labels = abs_._kmeans(data, k=3, metric="emd")
        assert len(labels) == 50
        assert all(0 <= l < 3 for l in labels)
