"""Reservoir sampling memory buffer for Deep CFR training samples."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    """A single training sample: features + regret/strategy targets."""
    features: np.ndarray
    targets: np.ndarray
    iteration: int


class ReservoirBuffer:
    """Reservoir sampling buffer — maintains uniform random sample of fixed size.

    As new samples arrive beyond capacity, each has a decreasing probability
    of replacing an existing sample, ensuring uniform sampling over the stream.
    """

    def __init__(self, capacity: int = 1_000_000, seed: int = 42) -> None:
        self.capacity = capacity
        self._buffer: list[Sample] = []
        self._count = 0
        self._rng = random.Random(seed)

    @property
    def size(self) -> int:
        return len(self._buffer)

    def add(self, features: np.ndarray, targets: np.ndarray, iteration: int) -> None:
        """Add a sample to the buffer using reservoir sampling."""
        sample = Sample(features=features, targets=targets, iteration=iteration)
        self._count += 1

        if len(self._buffer) < self.capacity:
            self._buffer.append(sample)
        else:
            # Replace with probability capacity / count
            idx = self._rng.randint(0, self._count - 1)
            if idx < self.capacity:
                self._buffer[idx] = sample

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random batch. Returns (features, targets) arrays."""
        n = min(batch_size, len(self._buffer))
        indices = self._rng.sample(range(len(self._buffer)), n)
        features = np.array([self._buffer[i].features for i in indices])
        targets = np.array([self._buffer[i].targets for i in indices])
        return features, targets

    def clear(self) -> None:
        self._buffer.clear()
        self._count = 0
