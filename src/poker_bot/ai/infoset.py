"""Information set storage for CFR."""

from __future__ import annotations

import numpy as np


class InfoSet:
    """Stores cumulative regret and cumulative strategy for one information set."""

    __slots__ = ("num_actions", "cumulative_regret", "cumulative_strategy", "reach_sum")

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.cumulative_regret = np.zeros(num_actions, dtype=np.float64)
        self.cumulative_strategy = np.zeros(num_actions, dtype=np.float64)
        self.reach_sum = 0.0

    def get_strategy(self) -> np.ndarray:
        """Get current strategy via regret matching."""
        positive = np.maximum(self.cumulative_regret, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(self.num_actions) / self.num_actions

    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy (converges to Nash equilibrium)."""
        total = self.cumulative_strategy.sum()
        if total > 0:
            return self.cumulative_strategy / total
        return np.ones(self.num_actions) / self.num_actions

    def update_strategy(self, reach_probability: float) -> np.ndarray:
        """Update cumulative strategy with reach probability. Returns current strategy."""
        strategy = self.get_strategy()
        self.cumulative_strategy += reach_probability * strategy
        self.reach_sum += reach_probability
        return strategy

    def update_regret(self, action_regrets: np.ndarray) -> None:
        """Add regrets to cumulative regret."""
        self.cumulative_regret += action_regrets

    def update_regret_cfr_plus(self, action_regrets: np.ndarray) -> None:
        """CFR+ regret update: floor cumulative regret at zero."""
        self.cumulative_regret = np.maximum(self.cumulative_regret + action_regrets, 0)

    def update_regret_dcfr(
        self, action_regrets: np.ndarray, t: int,
        alpha: float = 1.5, beta: float = 0.5, gamma: float = 2.0,
    ) -> None:
        """DCFR regret update: discount positive/negative regrets differently."""
        discount_pos = t**alpha / (t**alpha + 1)
        discount_neg = t**beta / (t**beta + 1)
        pos_mask = self.cumulative_regret > 0
        self.cumulative_regret[pos_mask] *= discount_pos
        self.cumulative_regret[~pos_mask] *= discount_neg
        self.cumulative_regret += action_regrets

    def update_strategy_dcfr(self, reach_probability: float, t: int, gamma: float = 2.0) -> np.ndarray:
        """DCFR strategy update: discount cumulative strategy before adding."""
        discount = (t / (t + 1)) ** gamma
        self.cumulative_strategy *= discount
        strategy = self.get_strategy()
        self.cumulative_strategy += reach_probability * strategy
        self.reach_sum += reach_probability
        return strategy


class InfoSetStore:
    """Dictionary-based store for information sets."""

    def __init__(self) -> None:
        self._store: dict[str, InfoSet] = {}

    def get_or_create(self, key: str, num_actions: int) -> InfoSet:
        if key not in self._store:
            self._store[key] = InfoSet(num_actions)
        return self._store[key]

    def __getitem__(self, key: str) -> InfoSet:
        return self._store[key]

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self):
        return iter(self._store)

    def keys(self):
        return self._store.keys()

    def items(self):
        return self._store.items()

    def values(self):
        return self._store.values()
