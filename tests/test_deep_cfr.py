"""Tests for Deep CFR — feature extraction, networks, memory, training loop."""

import numpy as np
import pytest

from poker_bot.ai.deep_cfr.features import (
    FEATURE_DIM,
    encode_card,
    encode_cards,
    extract_features,
)
from poker_bot.ai.deep_cfr.memory import ReservoirBuffer
from poker_bot.game.card import Card, Rank, Suit
from poker_bot.game.player import Player, PlayerStatus
from poker_bot.game.rules import BlindStructure
from poker_bot.game.state import GameState, Pot, Street

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestFeatureExtraction:
    def test_encode_card_dimensions(self):
        card = Card(Rank.ACE, Suit.SPADES)
        vec = encode_card(card)
        assert len(vec) == 17  # 13 ranks + 4 suits

    def test_encode_card_one_hot(self):
        card = Card(Rank.ACE, Suit.SPADES)
        vec = encode_card(card)
        assert vec.sum() == 2.0  # one rank + one suit

    def test_encode_cards_padding(self):
        cards = [Card(Rank.ACE, Suit.SPADES)]
        vec = encode_cards(cards, max_cards=3)
        assert len(vec) == 3 * 17
        # Second and third card slots should be zero
        assert vec[17:].sum() == 0.0

    def test_extract_features_dimensions(self):
        players = [
            Player(name="P0", seat=0, stack=9800),
            Player(name="P1", seat=1, stack=9700),
        ]
        players[0].hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        players[1].hole_cards = [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.JACK, Suit.CLUBS)]
        state = GameState(
            players=players,
            blinds=BlindStructure(50, 100),
            board=[Card(Rank.TEN, Suit.SPADES), Card(Rank.NINE, Suit.HEARTS), Card(Rank.EIGHT, Suit.DIAMONDS)],
            street=Street.FLOP,
            pots=[Pot(amount=500, eligible=[0, 1])],
            action_to=0,
        )
        features = extract_features(state, 0)
        assert len(features) == FEATURE_DIM
        assert features.dtype == np.float32

    def test_different_players_different_features(self):
        players = [
            Player(name="P0", seat=0, stack=9800),
            Player(name="P1", seat=1, stack=9700),
        ]
        players[0].hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        players[1].hole_cards = [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.JACK, Suit.CLUBS)]
        state = GameState(
            players=players,
            blinds=BlindStructure(50, 100),
            board=[],
            street=Street.PREFLOP,
            pots=[Pot(amount=150, eligible=[0, 1])],
            action_to=0,
        )
        f0 = extract_features(state, 0)
        f1 = extract_features(state, 1)
        assert not np.array_equal(f0, f1)


class TestReservoirBuffer:
    def test_add_within_capacity(self):
        buf = ReservoirBuffer(capacity=100)
        for i in range(50):
            buf.add(np.array([float(i)]), np.array([float(i)]), iteration=1)
        assert buf.size == 50

    def test_add_beyond_capacity(self):
        buf = ReservoirBuffer(capacity=10)
        for i in range(100):
            buf.add(np.array([float(i)]), np.array([float(i)]), iteration=1)
        assert buf.size == 10

    def test_sample_batch(self):
        buf = ReservoirBuffer(capacity=100)
        for i in range(50):
            buf.add(np.zeros(5), np.ones(3), iteration=1)
        features, targets = buf.sample_batch(10)
        assert features.shape == (10, 5)
        assert targets.shape == (10, 3)

    def test_sample_batch_clamps_to_size(self):
        buf = ReservoirBuffer(capacity=100)
        for i in range(5):
            buf.add(np.zeros(3), np.ones(2), iteration=1)
        features, targets = buf.sample_batch(20)
        assert features.shape == (5, 3)

    def test_clear(self):
        buf = ReservoirBuffer(capacity=100)
        for i in range(10):
            buf.add(np.zeros(3), np.ones(2), iteration=1)
        buf.clear()
        assert buf.size == 0

    def test_reservoir_property(self):
        """After adding many items, buffer stays at capacity with uniform sample."""
        buf = ReservoirBuffer(capacity=100, seed=42)
        for i in range(10000):
            buf.add(np.array([float(i)]), np.array([0.0]), iteration=1)
        assert buf.size == 100
        # Values should span the full range (not just first/last 100)
        values = [buf._buffer[i].features[0] for i in range(100)]
        assert max(values) > 5000  # should have some high values


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestNetworks:
    def test_value_network_forward(self):
        from poker_bot.ai.deep_cfr.networks import ValueNetwork
        net = ValueNetwork(input_dim=FEATURE_DIM, num_actions=9, hidden_dim=64)
        x = torch.randn(4, FEATURE_DIM)
        out = net(x)
        assert out.shape == (4, 9)

    def test_strategy_network_forward(self):
        from poker_bot.ai.deep_cfr.networks import StrategyNetwork
        net = StrategyNetwork(input_dim=FEATURE_DIM, num_actions=9, hidden_dim=64)
        x = torch.randn(4, FEATURE_DIM)
        out = net(x)
        assert out.shape == (4, 9)
        # Should sum to 1 (softmax)
        sums = out.sum(dim=1)
        np.testing.assert_allclose(sums.detach().numpy(), 1.0, atol=1e-5)

    def test_value_network_gradients(self):
        from poker_bot.ai.deep_cfr.networks import ValueNetwork
        net = ValueNetwork(input_dim=FEATURE_DIM, num_actions=9, hidden_dim=64)
        x = torch.randn(2, FEATURE_DIM)
        target = torch.randn(2, 9)
        out = net(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        # Check that gradients exist
        for param in net.parameters():
            assert param.grad is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestDeepCFRTraining:
    def test_training_loop_kuhn(self):
        """Deep CFR should run on Kuhn without crashing."""
        from poker_bot.ai.deep_cfr.deep_cfr import DeepCFR
        from poker_bot.games.kuhn import KuhnPoker

        kuhn = KuhnPoker()
        dcfr = DeepCFR(
            kuhn, num_actions=2, hidden_dim=32,
            buffer_capacity=10000, batch_size=64,
            train_steps=10,
        )
        # Run a few iterations
        dcfr.iterate(num_traversals=50)
        dcfr.iterate(num_traversals=50)

        # Should have collected samples
        assert dcfr._buffers[0].size > 0
        assert dcfr._buffers[1].size > 0

    def test_get_strategy_valid(self):
        """Strategy should be valid probability distribution."""
        from poker_bot.ai.deep_cfr.deep_cfr import DeepCFR
        from poker_bot.games.kuhn import KuhnPoker

        kuhn = KuhnPoker()
        dcfr = DeepCFR(
            kuhn, num_actions=2, hidden_dim=32,
            buffer_capacity=10000, batch_size=64,
            train_steps=10,
        )
        dcfr.iterate(num_traversals=20)

        state = kuhn.initial_state()
        # Apply chance to get to a player state
        outcomes = kuhn.chance_outcomes(state)
        for action, _ in outcomes:
            next_state = kuhn.apply_action(state, action)
            if kuhn.current_player(next_state) >= 0:
                actions = kuhn.legal_actions(next_state)
                strategy = dcfr.get_strategy(next_state, kuhn.current_player(next_state), actions)
                assert len(strategy) == len(actions)
                assert abs(strategy.sum() - 1.0) < 1e-5
                assert np.all(strategy >= 0)
                break
