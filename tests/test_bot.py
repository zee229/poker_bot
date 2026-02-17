"""Tests for BotPlayer."""

from poker_bot.ai.bot import BotPlayer
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.engine import GameEngine
from poker_bot.game.state import GameState


class TestBotPlayerEquityFallback:
    def _make_state(self) -> tuple[GameEngine, GameState]:
        engine = GameEngine(num_players=2, starting_stack=10000, seed=42)
        state = engine.new_hand()
        return engine, state

    def test_decide_returns_legal_action(self):
        engine, state = self._make_state()
        bot = BotPlayer(seed=42)
        actions = engine.get_legal_actions(state)
        chosen = bot.decide(state, state.action_to, actions)
        assert chosen.type in {a.type for a in actions}

    def test_decide_without_model(self):
        bot = BotPlayer(seed=42)
        assert not bot.has_model

    def test_decide_handles_empty_actions(self):
        bot = BotPlayer(seed=42)
        engine, state = self._make_state()
        action = bot.decide(state, state.action_to, [])
        assert action.type == ActionType.FOLD

    def test_decide_consistent_with_seed(self):
        engine, state = self._make_state()
        actions = engine.get_legal_actions(state)

        bot1 = BotPlayer(seed=123)
        bot2 = BotPlayer(seed=123)
        a1 = bot1.decide(state, state.action_to, actions)
        a2 = bot2.decide(state, state.action_to, actions)
        assert a1.type == a2.type

    def test_plays_full_hand_without_crash(self):
        engine = GameEngine(num_players=2, starting_stack=10000, seed=99)
        bot = BotPlayer(seed=42)
        state = engine.new_hand()

        for _ in range(500):  # safety limit (min-raise wars can be long)
            if state.hand_over:
                break
            actions = engine.get_legal_actions(state)
            if not actions:
                break
            action = bot.decide(state, state.action_to, actions)
            state = engine.apply_action(action, state)

        assert state.hand_over

    def test_plays_multiple_hands(self):
        engine = GameEngine(num_players=2, starting_stack=10000, seed=55)
        bot = BotPlayer(seed=42)

        for _ in range(3):
            engine.advance_dealer()
            state = engine.new_hand()
            for _ in range(500):
                if state.hand_over:
                    break
                actions = engine.get_legal_actions(state)
                if not actions:
                    break
                action = bot.decide(state, state.action_to, actions)
                state = engine.apply_action(action, state)
            assert state.hand_over
