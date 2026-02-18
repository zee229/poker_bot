"""Public Belief State (PBS) — core data structure for ReBeL."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poker_bot.game.state import GameState


@dataclass
class PublicBeliefState:
    """A Public Belief State encodes what's publicly known plus belief distributions.

    beliefs[player] is a probability distribution over that player's possible
    private hands, conditioned on the public action history.
    """

    beliefs: list[np.ndarray]
    public_state: GameState
    history: list[tuple[int, int]] = field(default_factory=list)

    @property
    def num_players(self) -> int:
        return len(self.beliefs)

    @property
    def num_hands(self) -> int:
        """Number of possible private hand buckets per player."""
        return len(self.beliefs[0]) if self.beliefs else 0

    def update_belief(
        self,
        player: int,
        action_idx: int,
        strategy_map: dict[int, np.ndarray],
    ) -> PublicBeliefState:
        """Bayesian belief update: P(hand|action) proportional to P(action|hand) * P(hand).

        strategy_map: hand_bucket -> strategy distribution over actions
        Returns a new PBS with updated beliefs.
        """
        new_beliefs = [b.copy() for b in self.beliefs]
        belief = new_beliefs[player]
        n_hands = len(belief)

        updated = np.zeros(n_hands, dtype=np.float64)
        for h in range(n_hands):
            if belief[h] <= 0:
                continue
            strat = strategy_map.get(h)
            if strat is None:
                # Unknown strategy for this hand — use uniform
                updated[h] = belief[h] / max(len(strat) if strat is not None else 1, 1)
                continue
            if action_idx < len(strat):
                updated[h] = belief[h] * strat[action_idx]
            # else: probability of this action from this hand is 0

        total = updated.sum()
        if total > 0:
            updated /= total
        else:
            # Fallback: if all beliefs are zero, keep uniform
            updated = np.ones(n_hands, dtype=np.float64) / n_hands

        new_beliefs[player] = updated
        new_history = self.history + [(player, action_idx)]

        return PublicBeliefState(
            beliefs=new_beliefs,
            public_state=self.public_state,
            history=new_history,
        )

    def to_features(self) -> np.ndarray:
        """Convert PBS to a feature vector for the value network.

        Feature layout:
        - Concatenated belief vectors: num_players * num_hands
        - Public state features: street (4), pot_ratio (1), num_active (1)
        Total: num_players * num_hands + 6
        """
        gs = self.public_state
        belief_feats = np.concatenate(self.beliefs)

        # Street one-hot (graceful fallback for non-GameState objects like KuhnState)
        street = np.zeros(4, dtype=np.float64)
        if hasattr(gs, 'street'):
            street_idx = min(int(gs.street), 3)
            street[street_idx] = 1.0

        # Pot and player count (graceful fallback)
        pot_ratio = 0.0
        num_active_ratio = 1.0
        if hasattr(gs, 'main_pot') and hasattr(gs, 'players'):
            total_stacks = sum(p.stack for p in gs.players)
            total_stacks = max(total_stacks, 1)
            pot_ratio = gs.main_pot / total_stacks
            num_active_ratio = gs.num_active / max(len(gs.players), 1)

        public_feats = np.array([pot_ratio, num_active_ratio], dtype=np.float64)

        return np.concatenate([belief_feats, street, public_feats])

    @staticmethod
    def feature_dim(num_players: int, num_hands: int) -> int:
        """Compute feature dimension for given configuration."""
        return num_players * num_hands + 4 + 2

    @staticmethod
    def initial(num_players: int, num_hands: int, public_state: GameState) -> PublicBeliefState:
        """Create initial PBS with uniform beliefs."""
        beliefs = [
            np.ones(num_hands, dtype=np.float64) / num_hands
            for _ in range(num_players)
        ]
        return PublicBeliefState(
            beliefs=beliefs,
            public_state=public_state,
            history=[],
        )
