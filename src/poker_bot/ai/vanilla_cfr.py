"""Vanilla CFR — full tree traversal, suitable for small games."""

from __future__ import annotations

import numpy as np

from poker_bot.ai.cfr_base import CFRBase, CFRVariant, GameAdapter


class VanillaCFR(CFRBase):
    """Vanilla Counterfactual Regret Minimization with full tree traversal."""

    def __init__(self, game: GameAdapter, variant: CFRVariant = CFRVariant.VANILLA) -> None:
        super().__init__(game)
        self.variant = variant

    def iterate(self) -> None:
        """Run one iteration: traverse for each player."""
        num_p = self.game.num_players()
        for player in range(num_p):
            reach = np.ones(num_p, dtype=np.float64)
            self._cfr(self.game.initial_state(), player, reach)
        self.iterations += 1

    def _cfr(self, state, traversing_player: int, reach: np.ndarray) -> float:
        """Recursive CFR traversal. Returns counterfactual value for traversing_player."""
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, traversing_player)

        current = self.game.current_player(state)

        if current == -1:  # chance node
            value = 0.0
            for action, prob in self.game.chance_outcomes(state):
                new_reach = reach.copy()
                # Chance doesn't belong to any player
                next_state = self.game.apply_action(state, action)
                value += prob * self._cfr(next_state, traversing_player, new_reach)
            return value

        actions = self.game.legal_actions(state)
        num_actions = len(actions)
        key = self.game.info_set_key(state, current)

        info_set = self.info_sets.get_or_create(key, num_actions)

        # Get strategy via regret matching
        if self.variant == CFRVariant.DCFR:
            strategy = info_set.update_strategy_dcfr(reach[current], self.iterations + 1)
        else:
            strategy = info_set.update_strategy(reach[current])

        action_values = np.zeros(num_actions, dtype=np.float64)
        node_value = 0.0

        for i, action in enumerate(actions):
            new_reach = reach.copy()
            new_reach[current] *= strategy[i]

            next_state = self.game.apply_action(state, action)
            action_values[i] = self._cfr(next_state, traversing_player, new_reach)
            node_value += strategy[i] * action_values[i]

        # Update regrets only for traversing player's info sets
        if current == traversing_player:
            # Counterfactual reach: product of opponents' reach probs
            cf_reach = np.prod(reach[:current]) * np.prod(reach[current + 1 :])
            regrets = cf_reach * (action_values - node_value)
            if self.variant == CFRVariant.CFR_PLUS:
                info_set.update_regret_cfr_plus(regrets)
            elif self.variant == CFRVariant.DCFR:
                info_set.update_regret_dcfr(regrets, self.iterations + 1)
            else:
                info_set.update_regret(regrets)

        return node_value
