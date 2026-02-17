"""Card abstraction via equity histograms and k-means clustering.

Bottom-up pipeline: river -> turn -> flop -> preflop.
Each street maps hands to buckets using equity-based features.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from poker_bot.game.card import Card, Rank, Suit, make_deck
from poker_bot.game.hand_eval import evaluate_hand


class CardAbstraction:
    """Manages card abstraction buckets for all streets."""

    def __init__(
        self,
        num_preflop_buckets: int = 8,
        num_flop_buckets: int = 50,
        num_turn_buckets: int = 50,
        num_river_buckets: int = 50,
        seed: int = 42,
    ) -> None:
        self.num_buckets = {
            "preflop": num_preflop_buckets,
            "flop": num_flop_buckets,
            "turn": num_turn_buckets,
            "river": num_river_buckets,
        }
        self.rng = np.random.RandomState(seed)
        self.centroids: dict[str, np.ndarray] = {}
        self._lut: dict[str, dict[str, int]] = {}

    def compute_river_buckets(
        self, num_samples: int = 10000
    ) -> dict[str, int]:
        """Compute river buckets based on hand strength (equity vs random)."""
        deck = make_deck()
        features = []
        keys = []

        rng = random.Random(42)

        for _ in range(num_samples):
            rng.shuffle(deck)
            hole = deck[:2]
            board = deck[2:7]

            result = evaluate_hand(hole + board)
            strength = result.score / (8 << 24)  # normalize

            key = _hand_key(hole, board)
            features.append([strength])
            keys.append(key)

        features_arr = np.array(features)
        labels = self._kmeans(features_arr, self.num_buckets["river"])

        bucket_map = {}
        for key, label in zip(keys, labels):
            bucket_map[key] = int(label)

        self._lut["river"] = bucket_map
        return bucket_map

    def compute_turn_buckets(
        self, num_samples: int = 5000, num_rollouts: int = 50
    ) -> dict[str, int]:
        """Compute turn buckets using equity distribution over river outcomes."""
        deck = make_deck()
        features = []
        keys = []
        num_river = self.num_buckets["river"]

        rng = random.Random(42)

        for _ in range(num_samples):
            rng.shuffle(deck)
            hole = deck[:2]
            board = deck[2:6]  # 4 community cards (turn)

            histogram = np.zeros(num_river)
            used = set(c.int_value for c in hole + board)
            remaining = [c for c in make_deck() if c.int_value not in used]

            river_lut = self._lut.get("river", {})
            for _ in range(num_rollouts):
                rng.shuffle(remaining)
                river_card = remaining[0]
                full_board = board + [river_card]
                river_key = _hand_key(hole, full_board)
                bucket = river_lut.get(
                    river_key, int(_fnv1a(river_key) % num_river)
                )
                histogram[bucket] += 1

            histogram /= histogram.sum() + 1e-10
            key = _hand_key(hole, board)
            features.append(histogram)
            keys.append(key)

        features_arr = np.array(features)
        labels = self._kmeans(features_arr, self.num_buckets["turn"], metric="emd")

        bucket_map = {}
        for key, label in zip(keys, labels):
            bucket_map[key] = int(label)

        self._lut["turn"] = bucket_map
        return bucket_map

    def compute_flop_buckets(
        self, num_samples: int = 5000, num_rollouts: int = 50
    ) -> dict[str, int]:
        """Compute flop buckets using equity distribution over turn outcomes."""
        deck = make_deck()
        features = []
        keys = []
        num_turn = self.num_buckets["turn"]

        rng = random.Random(42)

        for _ in range(num_samples):
            rng.shuffle(deck)
            hole = deck[:2]
            board = deck[2:5]  # 3 community cards (flop)

            histogram = np.zeros(num_turn)
            used = set(c.int_value for c in hole + board)
            remaining = [c for c in make_deck() if c.int_value not in used]

            turn_lut = self._lut.get("turn", {})
            for _ in range(num_rollouts):
                rng.shuffle(remaining)
                turn_card = remaining[0]
                turn_board = board + [turn_card]
                turn_key = _hand_key(hole, turn_board)
                bucket = turn_lut.get(
                    turn_key, int(_fnv1a(turn_key) % num_turn)
                )
                histogram[bucket] += 1

            histogram /= histogram.sum() + 1e-10
            key = _hand_key(hole, board)
            features.append(histogram)
            keys.append(key)

        features_arr = np.array(features)
        labels = self._kmeans(features_arr, self.num_buckets["flop"], metric="emd")

        bucket_map = {}
        for key, label in zip(keys, labels):
            bucket_map[key] = int(label)

        self._lut["flop"] = bucket_map
        return bucket_map

    def compute_preflop_buckets(self) -> dict[str, int]:
        """Compute preflop buckets based on hand equity estimates."""
        from poker_bot.ai.abstraction.isomorphism import preflop_hand_key

        deck = make_deck()
        features = []
        keys = []
        rng = random.Random(42)

        # Sample hands and estimate equity
        seen = set()
        for _ in range(10000):
            rng.shuffle(deck)
            hole = deck[:2]
            key = preflop_hand_key(hole[0], hole[1])
            if key in seen:
                continue
            seen.add(key)

            # Monte Carlo equity estimation
            wins = 0
            trials = 200
            for _ in range(trials):
                rng.shuffle(deck)
                board = deck[2:7]
                opp = deck[7:9]
                try:
                    my = evaluate_hand(hole + board)
                    their = evaluate_hand(opp + board)
                    if my.score > their.score:
                        wins += 1
                    elif my.score == their.score:
                        wins += 0.5
                except Exception:
                    continue

            equity = wins / trials
            features.append([equity])
            keys.append(key)

        features_arr = np.array(features)
        n_buckets = min(self.num_buckets["preflop"], len(features_arr))
        labels = self._kmeans(features_arr, n_buckets)

        bucket_map = {}
        for key, label in zip(keys, labels):
            bucket_map[key] = int(label)

        self._lut["preflop"] = bucket_map
        return bucket_map

    def get_bucket(self, street: str, hole_cards: list[Card], board: list[Card]) -> int:
        """Get bucket for a hand on a given street."""
        if street == "preflop":
            from poker_bot.ai.abstraction.isomorphism import preflop_hand_key
            key = preflop_hand_key(hole_cards[0], hole_cards[1])
        else:
            key = _hand_key(hole_cards, board)

        lut = self._lut.get(street, {})
        if key in lut:
            return lut[key]
        # Fallback: deterministic FNV-1a hash
        return int(_fnv1a(key) % self.num_buckets[street])

    def compute_all(
        self,
        river_samples: int = 10000,
        turn_samples: int = 5000,
        turn_rollouts: int = 50,
        flop_samples: int = 5000,
        flop_rollouts: int = 50,
    ) -> None:
        """Run full bottom-up pipeline: river -> turn -> flop -> preflop."""
        self.compute_river_buckets(num_samples=river_samples)
        self.compute_turn_buckets(num_samples=turn_samples, num_rollouts=turn_rollouts)
        self.compute_flop_buckets(num_samples=flop_samples, num_rollouts=flop_rollouts)
        self.compute_preflop_buckets()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for street, lut in self._lut.items():
            keys = list(lut.keys())
            values = [lut[k] for k in keys]
            np.save(path / f"{street}_keys.npy", np.array(keys))
            np.save(path / f"{street}_values.npy", np.array(values))

    def load(self, path: Path) -> None:
        for street in self.num_buckets:
            key_file = path / f"{street}_keys.npy"
            val_file = path / f"{street}_values.npy"
            if key_file.exists() and val_file.exists():
                keys = np.load(key_file, allow_pickle=True)
                values = np.load(val_file)
                self._lut[street] = dict(zip(keys, values.astype(int)))

    def _kmeans(
        self, data: np.ndarray, k: int, max_iter: int = 50, metric: str = "euclidean",
    ) -> np.ndarray:
        """K-means clustering with Euclidean or EMD distance."""
        n = len(data)
        if n <= k:
            return np.arange(n)

        # Random initialization
        indices = self.rng.choice(n, k, replace=False)
        centroids = data[indices].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            if metric == "emd":
                dists = _emd_distance_matrix(data, centroids)
            else:
                dists = cdist(data, centroids, metric="euclidean")
            new_labels = np.argmin(dists, axis=1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids[j] = data[mask].mean(axis=0)

        return labels


def _emd_distance_matrix(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute 1D EMD (Earth Mover's Distance) between data and centroids.

    For 1D histograms, EMD = sum(|CDF_a - CDF_b|). Vectorized with broadcasting.
    Returns shape (n, k) distance matrix.
    """
    cdf_data = np.cumsum(data, axis=1)          # (n, d)
    cdf_cent = np.cumsum(centroids, axis=1)      # (k, d)
    return np.sum(np.abs(cdf_data[:, None, :] - cdf_cent[None, :, :]), axis=2)


def _fnv1a(key: str) -> int:
    """Deterministic FNV-1a hash returning a plain int."""
    h = 14695981039346656037
    for byte in key.encode():
        h ^= byte
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _hand_key(hole: list[Card], board: list[Card]) -> str:
    """Create a string key for hole+board combination."""
    h = "".join(str(c) for c in sorted(hole, key=lambda c: (-c.rank, c.suit)))
    b = "".join(str(c) for c in sorted(board, key=lambda c: (-c.rank, c.suit)))
    return f"{h}|{b}"
