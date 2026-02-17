"""Deep CFR training loop — traverse, collect samples, retrain network."""

from __future__ import annotations

import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from poker_bot.ai.cfr_base import GameAdapter
from poker_bot.ai.deep_cfr.features import extract_features
from poker_bot.ai.deep_cfr.memory import ReservoirBuffer
from poker_bot.ai.deep_cfr.networks import ValueNetwork, _check_torch
from poker_bot.game.state import GameState


class DeepCFR:
    """Deep CFR solver using neural networks to generalize across info sets.

    Algorithm:
    1. Traverse game tree using current value network for strategy
    2. Collect (features, regrets) at each visited info set
    3. After N traversals, retrain value network on collected samples
    4. Repeat
    """

    def __init__(
        self,
        game: GameAdapter,
        num_actions: int = 9,
        hidden_dim: int = 256,
        buffer_capacity: int = 1_000_000,
        learning_rate: float = 1e-3,
        batch_size: int = 4096,
        train_steps: int = 4000,
        seed: int = 42,
    ) -> None:
        _check_torch()
        self.game = game
        self.num_actions = num_actions
        self.rng = random.Random(seed)
        self.iterations = 0

        # Per-player advantage memories and networks
        self._buffers: list[ReservoirBuffer] = []
        self._networks: list[ValueNetwork] = []
        self._optimizers: list = []

        for i in range(game.num_players()):
            self._buffers.append(ReservoirBuffer(capacity=buffer_capacity, seed=seed + i))
            net = ValueNetwork(num_actions=num_actions, hidden_dim=hidden_dim)
            self._networks.append(net)
            self._optimizers.append(optim.Adam(net.parameters(), lr=learning_rate))

        self.batch_size = batch_size
        self.train_steps = train_steps

    def iterate(self, num_traversals: int = 100) -> None:
        """Run one Deep CFR iteration: traverse then train."""
        self.iterations += 1

        for player in range(self.game.num_players()):
            for _ in range(num_traversals):
                self._traverse(self.game.initial_state(), player)

            # Retrain network for this player
            self._train_network(player)

    def _traverse(self, state, traversing_player: int) -> float:
        """External sampling traversal collecting advantage samples."""
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, traversing_player)

        current = self.game.current_player(state)

        if current == -1:  # chance node
            outcomes = self.game.chance_outcomes(state)
            probs = [p for _, p in outcomes]
            total = sum(probs)
            r = self.rng.random() * total
            cumsum = 0.0
            for action, prob in outcomes:
                cumsum += prob
                if r < cumsum:
                    next_state = self.game.apply_action(state, action)
                    return self._traverse(next_state, traversing_player)
            next_state = self.game.apply_action(state, outcomes[-1][0])
            return self._traverse(next_state, traversing_player)

        actions = self.game.legal_actions(state)
        num_actions = len(actions)

        # Get strategy from current player's value network
        strategy = self._predict_strategy(state, current, actions)

        if current == traversing_player:
            # Traverse all actions
            action_values = np.zeros(num_actions, dtype=np.float64)
            for i, action in enumerate(actions):
                next_state = self.game.apply_action(state, action)
                action_values[i] = self._traverse(next_state, traversing_player)

            node_value = np.dot(strategy[:num_actions], action_values)
            regrets = action_values - node_value

            # Store sample
            features = self._extract_features(state, current)
            # Pad regrets to num_actions size
            padded_regrets = np.zeros(self.num_actions, dtype=np.float32)
            padded_regrets[:num_actions] = regrets
            self._buffers[current].add(features, padded_regrets, self.iterations)

            return node_value
        else:
            # Sample opponent action
            probs = strategy[:num_actions]
            probs = probs / (probs.sum() + 1e-10)
            idx = self.rng.choices(range(num_actions), weights=probs)[0]
            next_state = self.game.apply_action(state, actions[idx])
            return self._traverse(next_state, traversing_player)

    def _predict_strategy(self, state, player: int, actions: list) -> np.ndarray:
        """Use value network to predict strategy via regret matching."""
        features = self._extract_features(state, player)
        net = self._networks[player]

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            predicted_regrets = net(x).squeeze(0).numpy()

        # Regret matching on predicted regrets
        num_actions = len(actions)
        regrets = predicted_regrets[:num_actions]
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(num_actions) / num_actions

    def _extract_features(self, state, player: int) -> np.ndarray:
        """Extract features — works for both toy games and NLHE."""
        gs = getattr(state, 'game_state', None)
        if gs is not None and isinstance(gs, GameState):
            return extract_features(gs, player)

        # Fallback for toy games: use info set key as hash
        key = self.game.info_set_key(state, player)
        # Simple hash-based feature encoding
        from poker_bot.ai.deep_cfr.features import FEATURE_DIM
        rng = np.random.RandomState(hash(key) & 0x7FFFFFFF)
        return rng.randn(FEATURE_DIM).astype(np.float32)

    def _train_network(self, player: int) -> float:
        """Train value network on collected advantage samples. Returns avg loss."""
        buffer = self._buffers[player]
        if buffer.size < self.batch_size:
            return 0.0

        net = self._networks[player]
        optimizer = self._optimizers[player]
        criterion = nn.MSELoss()

        net.train()
        total_loss = 0.0
        steps = min(self.train_steps, max(1, buffer.size // self.batch_size * 4))

        for _ in range(steps):
            features, targets = buffer.sample_batch(self.batch_size)
            x = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32)

            predicted = net(x)
            loss = criterion(predicted, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        net.eval()
        return total_loss / max(steps, 1)

    def get_strategy(self, state, player: int, actions: list) -> np.ndarray:
        """Get current strategy for a given state and player."""
        return self._predict_strategy(state, player, actions)
