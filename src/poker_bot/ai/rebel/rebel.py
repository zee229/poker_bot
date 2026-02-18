"""ReBeL training loop — recursive belief-based learning for poker.

Combines depth-limited CFR at Public Belief States (PBS) with a learned
value network for leaf evaluation. At each iteration:
1. Sample a random PBS
2. Run depth-limited CFR, using V(PBS) at leaf nodes
3. Use CFR result at root as training target for value network
4. Update value network via gradient descent
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from poker_bot.ai.cfr_base import CFRVariant, GameAdapter
from poker_bot.ai.infoset import InfoSetStore
from poker_bot.ai.rebel.belief import PublicBeliefState
from poker_bot.ai.vanilla_cfr import VanillaCFR
from poker_bot.game.state import GameState

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _PBSGameAdapter(GameAdapter):
    """Adapts a PBS-rooted game tree for depth-limited CFR.

    At leaf nodes (depth >= max_depth or terminal), uses the value network
    to estimate the counterfactual value instead of rolling out.
    """

    def __init__(
        self,
        pbs: PublicBeliefState,
        base_adapter: GameAdapter,
        value_fn: Callable[[PublicBeliefState], np.ndarray] | None = None,
        max_depth: int = 2,
    ) -> None:
        self._pbs = pbs
        self._base = base_adapter
        self._value_fn = value_fn
        self._max_depth = max_depth
        self._root_state = base_adapter.initial_state()

    def initial_state(self):
        return self._root_state

    def is_terminal(self, state) -> bool:
        if self._base.is_terminal(state):
            return True
        # Depth limit based on number of actions taken since root
        depth = getattr(state, '_depth', 0)
        return depth >= self._max_depth

    def terminal_utility(self, state, player: int) -> float:
        if self._base.is_terminal(state):
            return self._base.terminal_utility(state, player)

        # Use value network estimate at leaf
        if self._value_fn is not None:
            leaf_pbs = PublicBeliefState(
                beliefs=[b.copy() for b in self._pbs.beliefs],
                public_state=self._pbs.public_state,
                history=self._pbs.history,
            )
            values = self._value_fn(leaf_pbs)
            return float(values[player])

        # Fallback: fair share of pot
        return 0.0

    def current_player(self, state) -> int:
        return self._base.current_player(state)

    def num_players(self) -> int:
        return self._base.num_players()

    def info_set_key(self, state, player: int) -> str:
        return self._base.info_set_key(state, player)

    def legal_actions(self, state) -> list:
        return self._base.legal_actions(state)

    def apply_action(self, state, action):
        new_state = self._base.apply_action(state, action)
        # Track depth for leaf detection
        depth = getattr(state, '_depth', 0)
        try:
            new_state._depth = depth + 1
        except AttributeError:
            pass  # Some state objects don't allow attribute setting
        return new_state

    def chance_outcomes(self, state) -> list[tuple]:
        return self._base.chance_outcomes(state)


class ReBeL:
    """Recursive Belief-based Learning training loop.

    Training:
    1. Start from random PBS
    2. Run depth-limited CFR at PBS (like subgame solving)
    3. At leaf nodes: use value network V(PBS) instead of rollout
    4. Update value network to match CFR result at root
    5. Repeat

    Play:
    1. Maintain PBS (update beliefs as actions observed)
    2. At decision point: run CFR with V(PBS) at leaves
    3. Pick action from resulting strategy
    """

    def __init__(
        self,
        game_adapter: GameAdapter,
        num_hands: int,
        num_players: int = 2,
        cfr_iters: int = 100,
        cfr_depth: int = 2,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self._game = game_adapter
        self._num_hands = num_hands
        self._num_players = num_players
        self._cfr_iters = cfr_iters
        self._cfr_depth = cfr_depth
        self._rng = random.Random(seed)
        self._training_data: list[tuple[np.ndarray, np.ndarray]] = []

        if HAS_TORCH:
            from poker_bot.ai.rebel.value_net import PBSValueNetwork
            feat_dim = PublicBeliefState.feature_dim(num_players, num_hands)
            self._value_net = PBSValueNetwork(
                input_dim=feat_dim,
                num_players=num_players,
            )
            self._optimizer = optim.Adam(self._value_net.parameters(), lr=lr)
            self._loss_fn = nn.MSELoss()
        else:
            self._value_net = None
            self._optimizer = None
            self._loss_fn = None

    def _value_fn(self, pbs: PublicBeliefState) -> np.ndarray:
        """Evaluate a PBS using the value network."""
        if self._value_net is None:
            return np.zeros(self._num_players, dtype=np.float64)
        return self._value_net.predict(pbs.to_features())

    def _random_pbs(self, state: GameState) -> PublicBeliefState:
        """Create a random PBS from initial state with uniform beliefs."""
        return PublicBeliefState.initial(
            self._num_players, self._num_hands, state,
        )

    def train_step(self) -> float:
        """Run one ReBeL training step. Returns the loss value."""
        # 1. Create random PBS at initial state
        initial = self._game.initial_state()
        # We need a GameState for PBS — for toy games, initial_state might be the state directly
        gs = getattr(initial, 'game_state', None)
        if gs is None and isinstance(initial, GameState):
            gs = initial
        elif gs is None:
            # For toy games, create a dummy GameState-like PBS
            gs = initial

        pbs = self._random_pbs(gs)

        # 2. Run depth-limited CFR at this PBS
        pbs_adapter = _PBSGameAdapter(
            pbs=pbs,
            base_adapter=self._game,
            value_fn=self._value_fn if self._value_net else None,
            max_depth=self._cfr_depth,
        )
        cfr = VanillaCFR(pbs_adapter, variant=CFRVariant.CFR_PLUS)
        for _ in range(self._cfr_iters):
            cfr.iterate()

        # 3. Compute root value from CFR (expected value for each player)
        root_values = np.zeros(self._num_players, dtype=np.float64)
        for p in range(self._num_players):
            root_values[p] = self._compute_root_ev(cfr, pbs_adapter, p)

        # 4. Update value network
        if self._value_net is not None and HAS_TORCH:
            features = pbs.to_features()
            self._training_data.append((features, root_values))
            loss = self._update_value_net(features, root_values)
            return loss
        return 0.0

    def _compute_root_ev(
        self, cfr: VanillaCFR, adapter: GameAdapter, player: int,
    ) -> float:
        """Compute expected value at root for a player using average strategy."""
        return self._traverse_ev(
            cfr, adapter, adapter.initial_state(), player,
        )

    def _traverse_ev(
        self, cfr: VanillaCFR, adapter: GameAdapter,
        state, player: int,
    ) -> float:
        """Traverse the game tree computing EV under average strategy."""
        if adapter.is_terminal(state):
            return adapter.terminal_utility(state, player)

        current = adapter.current_player(state)
        if current == -1:
            value = 0.0
            for action, prob in adapter.chance_outcomes(state):
                next_state = adapter.apply_action(state, action)
                value += prob * self._traverse_ev(cfr, adapter, next_state, player)
            return value

        actions = adapter.legal_actions(state)
        key = adapter.info_set_key(state, current)

        if key in cfr.info_sets:
            strategy = cfr.info_sets[key].get_average_strategy()
        else:
            strategy = np.ones(len(actions)) / len(actions)

        value = 0.0
        for i, action in enumerate(actions):
            if i < len(strategy):
                next_state = adapter.apply_action(state, action)
                value += strategy[i] * self._traverse_ev(
                    cfr, adapter, next_state, player,
                )
        return value

    def _update_value_net(
        self, features: np.ndarray, target: np.ndarray,
    ) -> float:
        """Single gradient step on value network."""
        if not HAS_TORCH:
            return 0.0

        self._optimizer.zero_grad()
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        pred = self._value_net(x)
        loss = self._loss_fn(pred, y)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def train(self, num_steps: int) -> list[float]:
        """Run multiple training steps. Returns list of losses."""
        losses = []
        for _ in range(num_steps):
            loss = self.train_step()
            losses.append(loss)
        return losses

    def get_strategy(
        self, pbs: PublicBeliefState, player: int,
    ) -> dict[str, np.ndarray]:
        """Run CFR at the given PBS and return strategies.

        Used at play time to decide actions.
        """
        pbs_adapter = _PBSGameAdapter(
            pbs=pbs,
            base_adapter=self._game,
            value_fn=self._value_fn if self._value_net else None,
            max_depth=self._cfr_depth,
        )
        cfr = VanillaCFR(pbs_adapter, variant=CFRVariant.CFR_PLUS)
        for _ in range(self._cfr_iters):
            cfr.iterate()

        strategies = {}
        for key in cfr.info_sets:
            strategies[key] = cfr.info_sets[key].get_average_strategy()
        return strategies
