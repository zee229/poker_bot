"""Monte Carlo CFR with External Sampling."""

from __future__ import annotations

import random

import numpy as np

from poker_bot.ai.cfr_base import CFRBase, CFRVariant, GameAdapter


class MCCFR(CFRBase):
    """Monte Carlo CFR with External Sampling.

    Samples chance events and opponent actions, traverses all own actions.
    Does not require reach probabilities — simpler and scales to large games.
    """

    def __init__(
        self, game: GameAdapter, seed: int | None = None,
        variant: CFRVariant = CFRVariant.VANILLA,
    ) -> None:
        super().__init__(game)
        self.rng = random.Random(seed)
        self.variant = variant

    def iterate(self) -> None:
        """Run one iteration: traverse for each player."""
        for player in range(self.game.num_players()):
            self._traverse(self.game.initial_state(), player)
        self.iterations += 1

    def _traverse(self, state, traversing_player: int) -> float:
        """External sampling MCCFR traversal."""
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, traversing_player)

        current = self.game.current_player(state)

        if current == -1:  # chance node
            outcomes = self.game.chance_outcomes(state)
            probs = [p for _, p in outcomes]
            idx = self._sample_weighted(probs)
            action = outcomes[idx][0]
            next_state = self.game.apply_action(state, action)
            return self._traverse(next_state, traversing_player)

        actions = self.game.legal_actions(state)
        num_actions = len(actions)
        key = self.game.info_set_key(state, current)

        info_set = self.info_sets.get_or_create(key, num_actions)
        strategy = info_set.get_strategy()

        if current == traversing_player:
            # Traverse all actions, compute regrets
            action_values = np.zeros(num_actions, dtype=np.float64)
            for i, action in enumerate(actions):
                next_state = self.game.apply_action(state, action)
                action_values[i] = self._traverse(next_state, traversing_player)

            node_value = np.dot(strategy, action_values)

            # Update cumulative regret and strategy
            regrets = action_values - node_value
            if self.variant == CFRVariant.CFR_PLUS:
                info_set.update_regret_cfr_plus(regrets)
            elif self.variant == CFRVariant.DCFR:
                info_set.update_regret_dcfr(regrets, self.iterations + 1)
            else:
                info_set.update_regret(regrets)
            info_set.cumulative_strategy += strategy

            return node_value
        else:
            # Sample opponent action from current strategy
            idx = self._sample_weighted(strategy)
            next_state = self.game.apply_action(state, actions[idx])
            return self._traverse(next_state, traversing_player)

    def _sample_weighted(self, weights) -> int:
        """Sample index from weighted distribution using vectorized cumsum."""
        if not isinstance(weights, np.ndarray):
            weights = np.asarray(weights, dtype=np.float64)
        cumsum = np.cumsum(weights)
        return int(np.searchsorted(cumsum, self.rng.random()))
