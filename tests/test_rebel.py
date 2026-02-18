"""Tests for ReBeL (Recursive Belief-based Learning)."""

from __future__ import annotations

import numpy as np
import pytest

from poker_bot.ai.rebel.belief import PublicBeliefState
from poker_bot.games.kuhn import KuhnPoker, KuhnState

# Check if torch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestPublicBeliefState:
    def test_initial_uniform(self):
        """Initial PBS has uniform beliefs."""
        state = KuhnState(cards=("J", "Q"), history="")
        pbs = PublicBeliefState.initial(2, 3, state)
        assert pbs.num_players == 2
        assert pbs.num_hands == 3
        np.testing.assert_allclose(pbs.beliefs[0], [1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(pbs.beliefs[1], [1 / 3, 1 / 3, 1 / 3])

    def test_belief_update_valid_distribution(self):
        """After belief update, beliefs still sum to 1."""
        state = KuhnState(cards=("J", "Q"), history="")
        pbs = PublicBeliefState.initial(2, 3, state)

        # Strategy map: hand 0 always bets, hand 1 50/50, hand 2 never bets
        strategy_map = {
            0: np.array([0.0, 1.0]),  # always bet
            1: np.array([0.5, 0.5]),  # mixed
            2: np.array([1.0, 0.0]),  # always pass
        }

        updated = pbs.update_belief(0, action_idx=1, strategy_map=strategy_map)
        np.testing.assert_allclose(updated.beliefs[0].sum(), 1.0, atol=1e-10)
        # Player 1's beliefs should be unchanged
        np.testing.assert_allclose(updated.beliefs[1], pbs.beliefs[1])

    def test_belief_update_concentrates(self):
        """Belief update should concentrate on hands consistent with action."""
        state = KuhnState(cards=("J", "Q"), history="")
        pbs = PublicBeliefState.initial(2, 3, state)

        # hand 0 always bets, hand 1 and 2 never bet
        strategy_map = {
            0: np.array([0.0, 1.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([1.0, 0.0]),
        }

        updated = pbs.update_belief(0, action_idx=1, strategy_map=strategy_map)
        # Only hand 0 would take action 1 (bet), so belief should concentrate
        np.testing.assert_allclose(updated.beliefs[0][0], 1.0, atol=1e-10)

    def test_belief_update_preserves_zero(self):
        """Hands with zero prior stay at zero after update."""
        state = KuhnState(cards=("J", "Q"), history="")
        beliefs = [
            np.array([0.5, 0.5, 0.0]),  # hand 2 is impossible
            np.array([1 / 3, 1 / 3, 1 / 3]),
        ]
        pbs = PublicBeliefState(beliefs=beliefs, public_state=state)

        strategy_map = {
            0: np.array([0.5, 0.5]),
            1: np.array([0.5, 0.5]),
            2: np.array([0.5, 0.5]),
        }

        updated = pbs.update_belief(0, action_idx=0, strategy_map=strategy_map)
        assert updated.beliefs[0][2] == 0.0
        np.testing.assert_allclose(updated.beliefs[0].sum(), 1.0, atol=1e-10)

    def test_history_appended(self):
        """Belief update appends to history."""
        state = KuhnState(cards=("J", "Q"), history="")
        pbs = PublicBeliefState.initial(2, 3, state)
        strategy_map = {i: np.array([0.5, 0.5]) for i in range(3)}
        updated = pbs.update_belief(0, 1, strategy_map)
        assert updated.history == [(0, 1)]
        updated2 = updated.update_belief(1, 0, strategy_map)
        assert updated2.history == [(0, 1), (1, 0)]

    def test_to_features_shape(self):
        """PBS feature vector has correct dimension."""
        state = KuhnState(cards=("J", "Q"), history="")
        pbs = PublicBeliefState.initial(2, 3, state)
        feats = pbs.to_features()
        expected_dim = PublicBeliefState.feature_dim(2, 3)
        assert len(feats) == expected_dim

    def test_feature_dim_formula(self):
        """Feature dim = num_players * num_hands + 6."""
        assert PublicBeliefState.feature_dim(2, 3) == 2 * 3 + 4 + 2  # 12
        assert PublicBeliefState.feature_dim(2, 10) == 2 * 10 + 6  # 26
        assert PublicBeliefState.feature_dim(6, 50) == 6 * 50 + 6  # 306


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPBSValueNetwork:
    def test_forward_shape(self):
        """Value network output has shape (batch, num_players)."""
        from poker_bot.ai.rebel.value_net import PBSValueNetwork
        feat_dim = PublicBeliefState.feature_dim(2, 3)
        net = PBSValueNetwork(input_dim=feat_dim, num_players=2)
        x = torch.randn(4, feat_dim)
        out = net(x)
        assert out.shape == (4, 2)

    def test_predict_returns_numpy(self):
        """predict() returns numpy array of correct shape."""
        from poker_bot.ai.rebel.value_net import PBSValueNetwork
        feat_dim = PublicBeliefState.feature_dim(2, 3)
        net = PBSValueNetwork(input_dim=feat_dim, num_players=2)
        feats = np.random.randn(feat_dim).astype(np.float32)
        values = net.predict(feats)
        assert isinstance(values, np.ndarray)
        assert values.shape == (2,)

    def test_single_player(self):
        from poker_bot.ai.rebel.value_net import PBSValueNetwork
        feat_dim = 10
        net = PBSValueNetwork(input_dim=feat_dim, num_players=1)
        x = torch.randn(2, feat_dim)
        out = net(x)
        assert out.shape == (2, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestReBeLTraining:
    def test_train_step_runs(self):
        """Single training step completes without error on Kuhn."""
        from poker_bot.ai.rebel.rebel import ReBeL
        game = KuhnPoker()
        rebel = ReBeL(
            game_adapter=game,
            num_hands=3,
            num_players=2,
            cfr_iters=10,
            cfr_depth=4,  # deep enough for Kuhn
        )
        loss = rebel.train_step()
        assert isinstance(loss, float)

    def test_train_multiple_steps(self):
        """Multiple training steps run and produce losses."""
        from poker_bot.ai.rebel.rebel import ReBeL
        game = KuhnPoker()
        rebel = ReBeL(
            game_adapter=game,
            num_hands=3,
            num_players=2,
            cfr_iters=10,
            cfr_depth=4,
        )
        losses = rebel.train(5)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)

    def test_get_strategy_returns_dict(self):
        """get_strategy returns non-empty strategy dict."""
        from poker_bot.ai.rebel.rebel import ReBeL
        game = KuhnPoker()
        rebel = ReBeL(
            game_adapter=game,
            num_hands=3,
            num_players=2,
            cfr_iters=20,
            cfr_depth=4,
        )
        state = KuhnState(cards=("J", "Q"), history="")
        pbs = PublicBeliefState.initial(2, 3, state)
        strategies = rebel.get_strategy(pbs, player=0)
        assert len(strategies) > 0
        for key, strat in strategies.items():
            np.testing.assert_allclose(strat.sum(), 1.0, atol=1e-6)


class TestReBeLNoTorch:
    """Test that ReBeL degrades gracefully without PyTorch."""

    def test_belief_state_no_torch(self):
        """PBS works without torch."""
        state = KuhnState(cards=("J", "Q"), history="")
        pbs = PublicBeliefState.initial(2, 3, state)
        feats = pbs.to_features()
        assert len(feats) > 0

    def test_belief_update_no_torch(self):
        """Belief update works without torch."""
        state = KuhnState(cards=("J", "Q"), history="")
        pbs = PublicBeliefState.initial(2, 3, state)
        strategy_map = {i: np.array([0.5, 0.5]) for i in range(3)}
        updated = pbs.update_belief(0, 1, strategy_map)
        np.testing.assert_allclose(updated.beliefs[0].sum(), 1.0)
