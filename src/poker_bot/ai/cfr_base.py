"""Abstract base class for CFR solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from poker_bot.ai.infoset import InfoSetStore


class GameAdapter(ABC):
    """Abstract interface for games compatible with CFR."""

    @abstractmethod
    def initial_state(self):
        """Return the initial game state."""

    @abstractmethod
    def is_terminal(self, state) -> bool:
        """Check if state is terminal."""

    @abstractmethod
    def terminal_utility(self, state, player: int) -> float:
        """Return utility for player at a terminal state."""

    @abstractmethod
    def current_player(self, state) -> int:
        """Return the player whose turn it is. -1 for chance."""

    @abstractmethod
    def num_players(self) -> int:
        """Return number of players."""

    @abstractmethod
    def info_set_key(self, state, player: int) -> str:
        """Return information set key for the given player at this state."""

    @abstractmethod
    def legal_actions(self, state) -> list:
        """Return list of legal actions."""

    @abstractmethod
    def apply_action(self, state, action):
        """Apply action and return new state (should not mutate input)."""

    @abstractmethod
    def chance_outcomes(self, state) -> list[tuple]:
        """Return list of (action, probability) for chance nodes."""


class CFRBase(ABC):
    """Base class for CFR algorithms."""

    def __init__(self, game: GameAdapter) -> None:
        self.game = game
        self.info_sets = InfoSetStore()
        self.iterations = 0

    @abstractmethod
    def iterate(self) -> None:
        """Run one iteration of CFR."""

    def compute_exploitability(self) -> float:
        """Compute exploitability of average strategy via info-set-constrained best response."""
        total = 0.0
        for p in range(self.game.num_players()):
            # Pass 1: collect weighted action values per info set
            info_set_av: dict[str, list[float]] = {}
            self._br_collect(self.game.initial_state(), p, 1.0, info_set_av)

            # Determine best action at each info set
            br_actions: dict[str, int] = {}
            for key, values in info_set_av.items():
                br_actions[key] = int(max(range(len(values)), key=lambda i: values[i]))

            # Pass 2: evaluate the constrained BR strategy
            total += self._br_evaluate(self.game.initial_state(), p, br_actions)
        return total

    def _br_collect(
        self, state, br_player: int, opp_reach: float, info_set_av: dict
    ) -> float:
        """Collect weighted action values at each br_player info set.

        Returns subtree value assuming locally optimal play (used for weighting).
        """
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, br_player)

        current = self.game.current_player(state)

        if current == -1:
            value = 0.0
            for action, prob in self.game.chance_outcomes(state):
                next_state = self.game.apply_action(state, action)
                value += prob * self._br_collect(next_state, br_player, opp_reach * prob, info_set_av)
            return value

        actions = self.game.legal_actions(state)

        if current == br_player:
            key = self.game.info_set_key(state, current)
            if key not in info_set_av:
                info_set_av[key] = [0.0] * len(actions)

            action_values = []
            for i, action in enumerate(actions):
                next_state = self.game.apply_action(state, action)
                v = self._br_collect(next_state, br_player, opp_reach, info_set_av)
                info_set_av[key][i] += opp_reach * v
                action_values.append(v)

            return max(action_values)
        else:
            key = self.game.info_set_key(state, current)
            if key in self.info_sets:
                strategy = self.info_sets[key].get_average_strategy()
            else:
                strategy = [1.0 / len(actions)] * len(actions)

            value = 0.0
            for i, action in enumerate(actions):
                next_state = self.game.apply_action(state, action)
                value += strategy[i] * self._br_collect(
                    next_state, br_player, opp_reach * strategy[i], info_set_av
                )
            return value

    def _br_evaluate(self, state, br_player: int, br_actions: dict[str, int]) -> float:
        """Evaluate constrained best response strategy value."""
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, br_player)

        current = self.game.current_player(state)

        if current == -1:
            value = 0.0
            for action, prob in self.game.chance_outcomes(state):
                next_state = self.game.apply_action(state, action)
                value += prob * self._br_evaluate(next_state, br_player, br_actions)
            return value

        actions = self.game.legal_actions(state)

        if current == br_player:
            key = self.game.info_set_key(state, current)
            best_idx = br_actions.get(key, 0)
            next_state = self.game.apply_action(state, actions[best_idx])
            return self._br_evaluate(next_state, br_player, br_actions)
        else:
            key = self.game.info_set_key(state, current)
            if key in self.info_sets:
                strategy = self.info_sets[key].get_average_strategy()
            else:
                strategy = [1.0 / len(actions)] * len(actions)

            value = 0.0
            for i, action in enumerate(actions):
                next_state = self.game.apply_action(state, action)
                value += strategy[i] * self._br_evaluate(next_state, br_player, br_actions)
            return value
