"""Value network for PBS evaluation in ReBeL."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for ReBeL. "
            "Install with: uv pip install 'poker-bot[deep]'"
        )


class PBSValueNetwork(nn.Module if HAS_TORCH else object):
    """Maps PBS features to expected value per player.

    Input: PBS feature vector (beliefs + public state features)
    Output: value estimate per player (num_players,)
    """

    def __init__(
        self,
        input_dim: int,
        num_players: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        _check_torch()
        super().__init__()
        self.num_players = num_players
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_players),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, input_dim) -> (batch, num_players)."""
        return self.net(x)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict values for a single PBS feature vector. Returns (num_players,) array."""
        _check_torch()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            out = self.forward(x)
            return out.squeeze(0).numpy()
