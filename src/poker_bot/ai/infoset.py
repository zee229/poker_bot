"""Information set storage for CFR."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    @njit(cache=True)
    def _regret_match(cumulative_regret: np.ndarray) -> np.ndarray:
        """Numba-accelerated regret matching."""
        n = len(cumulative_regret)
        strategy = np.empty(n, dtype=np.float64)
        pos_sum = 0.0
        for i in range(n):
            s = max(cumulative_regret[i], 0.0)
            strategy[i] = s
            pos_sum += s
        if pos_sum > 0:
            for i in range(n):
                strategy[i] /= pos_sum
        else:
            inv_n = 1.0 / n
            for i in range(n):
                strategy[i] = inv_n
        return strategy

    @njit(cache=True)
    def _regret_update_cfr_plus(cumulative_regret: np.ndarray, action_regrets: np.ndarray) -> np.ndarray:
        """Numba-accelerated CFR+ regret update."""
        n = len(cumulative_regret)
        result = np.empty(n, dtype=np.float64)
        for i in range(n):
            v = cumulative_regret[i] + action_regrets[i]
            result[i] = v if v > 0.0 else 0.0
        return result

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def _regret_match(cumulative_regret: np.ndarray) -> np.ndarray:
        """Numpy fallback regret matching."""
        positive = np.maximum(cumulative_regret, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.full(len(cumulative_regret), 1.0 / len(cumulative_regret))


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
        return _regret_match(self.cumulative_regret)

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
        if _HAS_NUMBA:
            self.cumulative_regret = _regret_update_cfr_plus(self.cumulative_regret, action_regrets)
        else:
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
