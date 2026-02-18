"""Strategy storage using numpy memmap for large-scale CFR strategies."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class StrategyStore:
    """Stores CFR strategies in numpy arrays for fast lookup.

    Uses sorted uint64 hashes with binary search for O(log n) lookup.
    """

    def __init__(self, max_actions: int = 9) -> None:
        self.max_actions = max_actions
        self._keys: np.ndarray | None = None
        self._strategies: np.ndarray | None = None

    def from_info_sets(self, info_sets) -> None:
        """Build strategy store from InfoSetStore."""
        items = list(info_sets.items())
        n = len(items)
        keys = np.zeros(n, dtype=np.uint64)
        strategies = np.zeros((n, self.max_actions), dtype=np.float32)

        for i, (key, info_set) in enumerate(items):
            keys[i] = _hash_key(key)
            avg = info_set.get_average_strategy()
            strategies[i, : len(avg)] = avg

        # Sort by key for binary search
        order = np.argsort(keys)
        self._keys = keys[order]
        self._strategies = strategies[order]

    def lookup(self, key: str) -> np.ndarray | None:
        """Look up strategy for an info set key. Returns None if not found."""
        if self._keys is None:
            return None
        h = _hash_key(key)
        idx = np.searchsorted(self._keys, h)
        if idx < len(self._keys) and self._keys[idx] == h:
            return self._strategies[idx]
        return None

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._keys is not None:
            np.save(path / "keys.npy", self._keys)
            np.save(path / "strategy.npy", self._strategies)

    def load(self, path: Path) -> None:
        key_file = path / "keys.npy"
        strat_file = path / "strategy.npy"
        if key_file.exists() and strat_file.exists():
            self._keys = np.load(key_file)
            self._strategies = np.load(strat_file)

    @property
    def size(self) -> int:
        return len(self._keys) if self._keys is not None else 0


def _hash_key(key: str) -> np.uint64:
    """Hash a string key to uint64 using FNV-1a (overflow is intentional)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        h = np.uint64(14695981039346656037)
        for byte in key.encode():
            h ^= np.uint64(byte)
            h *= np.uint64(1099511628211)
    return h
