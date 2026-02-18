"""Tests for opponent modeling."""

from __future__ import annotations

import numpy as np
import pytest

from poker_bot.ai.opponent_model import OpponentModel, OpponentStats, PlayerType
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.card import Card
from poker_bot.game.player import Player, PlayerStatus
from poker_bot.game.rules import BlindStructure
from poker_bot.game.state import GameState, Pot, Street


def _make_state(
    street: Street = Street.PREFLOP,
    pot: int = 150,
    current_bet: int = 100,
    action_history: list[list[tuple[int, Action]]] | None = None,
) -> GameState:
    blinds = BlindStructure(50, 100)
    p0 = Player(seat=0, stack=5000, name="P0")
    p0.status = PlayerStatus.ACTIVE
    p0.bet_this_street = 50
    p1 = Player(seat=1, stack=5000, name="P1")
    p1.status = PlayerStatus.ACTIVE
    p1.bet_this_street = 100
    return GameState(
        players=[p0, p1],
        blinds=blinds,
        board=[Card.from_str("Ah"), Card.from_str("Kd"), Card.from_str("Tc")] if street != Street.PREFLOP else [],
        street=street,
        pots=[Pot(amount=pot, eligible=[0, 1])],
        current_bet=current_bet,
        min_raise=100,
        action_to=0,
        action_history=action_history or [[], [], [], []],
    )


class TestOpponentStats:
    def test_initial_stats(self):
        stats = OpponentStats()
        assert stats.hands_seen == 0
        assert stats.vpip_pct == 0.0
        assert stats.pfr_pct == 0.0
        assert stats.aggression_factor == 0.0

    def test_vpip_tracking(self):
        stats = OpponentStats()
        stats.start_hand()
        stats._hand_vpip = True
        stats.end_hand()
        assert stats.vpip_pct == 1.0

        stats.start_hand()
        stats.end_hand()
        assert stats.vpip_pct == 0.5

    def test_pfr_tracking(self):
        stats = OpponentStats()
        stats.start_hand()
        stats._hand_pfr = True
        stats.end_hand()
        assert stats.pfr_pct == 1.0

    def test_aggression_factor(self):
        stats = OpponentStats()
        stats.af_bets = 6
        stats.af_calls = 3
        assert stats.aggression_factor == 2.0

    def test_aggression_factor_no_calls(self):
        stats = OpponentStats()
        stats.af_bets = 5
        assert stats.aggression_factor == 5.0

    def test_cbet_pct(self):
        stats = OpponentStats()
        stats.cbet_opportunities = 4
        stats.cbet_made = 3
        assert stats.cbet_pct == 0.75


class TestOpponentModelObserve:
    def test_observe_raise_preflop(self):
        model = OpponentModel()
        state = _make_state(street=Street.PREFLOP)
        model.start_hand([0, 1])
        model.observe_action(0, Action.raise_to(300), state)
        stats = model.get_stats(0)
        assert stats.af_bets == 1
        assert stats._hand_vpip is True
        assert stats._hand_pfr is True

    def test_observe_call_preflop(self):
        model = OpponentModel()
        state = _make_state(street=Street.PREFLOP)
        model.start_hand([0, 1])
        model.observe_action(0, Action.call(100), state)
        stats = model.get_stats(0)
        assert stats.af_calls == 1
        assert stats._hand_vpip is True
        assert stats._hand_pfr is False

    def test_observe_fold(self):
        model = OpponentModel()
        state = _make_state(street=Street.PREFLOP)
        model.start_hand([0, 1])
        model.observe_action(0, Action.fold(), state)
        stats = model.get_stats(0)
        assert stats.af_bets == 0
        assert stats.af_calls == 0
        assert stats._hand_vpip is False

    def test_stats_accumulate_across_hands(self):
        model = OpponentModel()
        state = _make_state(street=Street.PREFLOP)

        # Hand 1: raise
        model.start_hand([0])
        model.observe_action(0, Action.raise_to(300), state)
        model.end_hand([0])

        # Hand 2: call
        model.start_hand([0])
        model.observe_action(0, Action.call(100), state)
        model.end_hand([0])

        # Hand 3: fold
        model.start_hand([0])
        model.observe_action(0, Action.fold(), state)
        model.end_hand([0])

        stats = model.get_stats(0)
        assert stats.hands_seen == 3
        assert stats.vpip == 2  # raise + call
        assert stats.pfr == 1  # only the raise


class TestPlayerClassification:
    def _make_model_with_stats(
        self, vpip_pct: float, pfr_pct: float, af: float,
    ) -> OpponentModel:
        """Create a model with synthetic stats for seat 0."""
        model = OpponentModel()
        stats = model.get_stats(0)
        n = 100
        stats.hands_seen = n
        stats.vpip = int(vpip_pct * n)
        stats.pfr = int(pfr_pct * n)
        if af >= 1:
            stats.af_bets = int(af * 10)
            stats.af_calls = 10
        else:
            stats.af_bets = int(af * 10)
            stats.af_calls = 10
        return model

    def test_unknown_few_hands(self):
        model = OpponentModel()
        stats = model.get_stats(0)
        stats.hands_seen = 10
        assert model.classify(0) == PlayerType.UNKNOWN

    def test_nit(self):
        model = self._make_model_with_stats(vpip_pct=0.12, pfr_pct=0.08, af=2.0)
        assert model.classify(0) == PlayerType.NIT

    def test_tag(self):
        model = self._make_model_with_stats(vpip_pct=0.22, pfr_pct=0.18, af=2.5)
        assert model.classify(0) == PlayerType.TAG

    def test_lag(self):
        model = self._make_model_with_stats(vpip_pct=0.35, pfr_pct=0.25, af=3.0)
        assert model.classify(0) == PlayerType.LAG

    def test_calling_station(self):
        model = self._make_model_with_stats(vpip_pct=0.45, pfr_pct=0.10, af=0.8)
        assert model.classify(0) == PlayerType.CALLING_STATION

    def test_maniac(self):
        model = self._make_model_with_stats(vpip_pct=0.60, pfr_pct=0.40, af=4.0)
        assert model.classify(0) == PlayerType.MANIAC


class TestExploitativeAdjustments:
    def test_calling_station_less_bluff(self):
        model = OpponentModel()
        stats = model.get_stats(0)
        stats.hands_seen = 100
        stats.vpip = 45
        stats.af_bets = 8
        stats.af_calls = 10
        assert model.classify(0) == PlayerType.CALLING_STATION

        adj = model.get_exploitative_adjustments(0)
        assert adj["bluff_factor"] < 1.0
        assert adj["value_bet_factor"] > 1.0

    def test_nit_more_stealing(self):
        model = OpponentModel()
        stats = model.get_stats(0)
        stats.hands_seen = 100
        stats.vpip = 12
        stats.pfr = 8
        stats.af_bets = 20
        stats.af_calls = 10
        assert model.classify(0) == PlayerType.NIT

        adj = model.get_exploitative_adjustments(0)
        assert adj["steal_factor"] > 1.0
        assert adj["call_factor"] < 1.0

    def test_unknown_no_adjustment(self):
        model = OpponentModel()
        adj = model.get_exploitative_adjustments(0)
        assert adj["value_bet_factor"] == 1.0
        assert adj["bluff_factor"] == 1.0
        assert adj["call_factor"] == 1.0


class TestStrategyAdjustment:
    def test_adjust_strategy_unknown_unchanged(self):
        model = OpponentModel()
        strategy = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        adjusted = model.adjust_strategy(strategy, 0, _make_state())
        np.testing.assert_allclose(adjusted, strategy)

    def test_adjust_strategy_calling_station(self):
        model = OpponentModel()
        stats = model.get_stats(0)
        stats.hands_seen = 100
        stats.vpip = 45
        stats.af_bets = 8
        stats.af_calls = 10
        assert model.classify(0) == PlayerType.CALLING_STATION

        strategy = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        adjusted = model.adjust_strategy(strategy, 0, _make_state())
        # Should be normalized
        np.testing.assert_allclose(adjusted.sum(), 1.0)
        # Strategy should differ from original
        assert not np.allclose(adjusted, strategy)

    def test_adjust_strategy_preserves_normalization(self):
        model = OpponentModel()
        stats = model.get_stats(0)
        stats.hands_seen = 100
        stats.vpip = 60
        stats.pfr = 40
        stats.af_bets = 40
        stats.af_calls = 10
        strategy = np.array([0.05, 0.1, 0.15, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02])
        adjusted = model.adjust_strategy(strategy, 0, _make_state())
        np.testing.assert_allclose(adjusted.sum(), 1.0, atol=1e-10)


class TestBotPlayerWithOpponentModel:
    def test_bot_with_opponent_model_no_crash(self):
        """BotPlayer with opponent model should not crash during equity fallback."""
        from poker_bot.ai.bot import BotPlayer
        from poker_bot.game.engine import GameEngine

        model = OpponentModel()
        bot = BotPlayer(opponent_model=model)

        engine = GameEngine(num_players=2, seed=42)
        state = engine.new_hand()

        model.start_hand([0, 1])

        while not state.hand_over:
            actions = engine.get_legal_actions(state)
            if not actions:
                break
            action = bot.decide(state, state.action_to, actions)
            model.observe_action(state.action_to, action, state)
            state = engine.apply_action(action, state)

        model.end_hand([0, 1])
