"""Neural network architectures for Deep CFR."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from poker_bot.ai.deep_cfr.features import FEATURE_DIM


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for Deep CFR. "
            "Install with: uv pip install 'poker-bot[deep]'"
        )


class ValueNetwork(nn.Module if HAS_TORCH else object):
    """MLP that predicts regret values for each action given game state features.

    Architecture: input -> FC -> ReLU -> FC -> ReLU -> FC -> output
    """

    def __init__(self, input_dim: int = FEATURE_DIM, num_actions: int = 9, hidden_dim: int = 256):
        _check_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class StrategyNetwork(nn.Module if HAS_TORCH else object):
    """MLP that predicts strategy (action probabilities) from game state features.

    Architecture: input -> FC -> ReLU -> FC -> ReLU -> FC -> Softmax
    """

    def __init__(self, input_dim: int = FEATURE_DIM, num_actions: int = 9, hidden_dim: int = 256):
        _check_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
