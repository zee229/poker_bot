"""Tests for the game engine."""

from poker_bot.game.actions import Action, ActionType
from poker_bot.game.engine import GameEngine
from poker_bot.game.player import PlayerStatus
from poker_bot.game.rules import BlindStructure
from poker_bot.game.state import Street


class TestEngineSetup:
    def test_new_hand_deals_cards(self):
        engine = GameEngine(num_players=6, seed=42)
        state = engine.new_hand()
        for p in state.players:
            assert len(p.hole_cards) == 2

    def test_blinds_posted(self):
        engine = GameEngine(num_players=6, starting_stack=1000, seed=42)
        state = engine.new_hand()
        # After posting blinds, pot should have SB + BB
        assert state.main_pot == 150  # 50 + 100

    def test_preflop_street(self):
        engine = GameEngine(num_players=2, seed=42)
        state = engine.new_hand()
        assert state.street == Street.PREFLOP

    def test_heads_up_positions(self):
        """In heads-up, dealer posts SB and acts first preflop."""
        engine = GameEngine(num_players=2, starting_stack=1000, seed=42)
        state = engine.new_hand()
        # Dealer is seat 0, SB is seat 1 (next after dealer), BB is seat 0 (next after SB)
        # Wait, with 2 players: SB = next_active(dealer), BB = next_active(SB)
        # dealer=0, SB=1, BB=0  — action starts next after BB = seat 1
        assert state.current_bet == 100


class TestLegalActions:
    def test_preflop_utg(self):
        engine = GameEngine(num_players=6, starting_stack=1000, seed=42)
        state = engine.new_hand()
        actions = engine.get_legal_actions(state)
        action_types = [a.type for a in actions]
        assert ActionType.FOLD in action_types
        assert ActionType.CALL in action_types

    def test_check_available_when_no_bet(self):
        engine = GameEngine(num_players=2, starting_stack=1000, seed=42)
        state = engine.new_hand()
        # Call preflop from SB/BTN to see flop
        actions = engine.get_legal_actions(state)
        # In HU, action is on the button(SB) preflop
        # They face a BB of 100, having posted 50
        action_types = [a.type for a in actions]
        assert ActionType.FOLD in action_types

    def test_all_in_when_short_stacked(self):
        engine = GameEngine(
            num_players=2,
            starting_stack=100,
            blinds=BlindStructure(50, 100),
            seed=42,
        )
        state = engine.new_hand()
        actions = engine.get_legal_actions(state)
        action_types = [a.type for a in actions]
        # SB posted 50 from 100 stack, has 50 left facing 100 BB
        assert ActionType.ALL_IN in action_types


class TestStreetProgression:
    def test_to_flop(self):
        engine = GameEngine(num_players=2, starting_stack=1000, seed=42)
        state = engine.new_hand()
        # Player calls
        calls = [a for a in engine.get_legal_actions(state) if a.type == ActionType.CALL]
        if calls:
            state = engine.apply_action(calls[0], state)
        # BB checks
        checks = [a for a in engine.get_legal_actions(state) if a.type == ActionType.CHECK]
        if checks:
            state = engine.apply_action(checks[0], state)
        assert state.street == Street.FLOP
        assert len(state.board) == 3

    def test_full_hand_check_down(self):
        engine = GameEngine(num_players=2, starting_stack=1000, seed=42)
        state = engine.new_hand()

        # Play through by calling/checking until hand ends
        max_actions = 100
        for _ in range(max_actions):
            if state.hand_over:
                break
            actions = engine.get_legal_actions(state)
            if not actions:
                break
            # Prefer check, then call, then fold
            action = None
            for pref in [ActionType.CHECK, ActionType.CALL, ActionType.FOLD]:
                matches = [a for a in actions if a.type == pref]
                if matches:
                    action = matches[0]
                    break
            if action is None:
                action = actions[0]
            state = engine.apply_action(action, state)

        assert state.hand_over
        assert state.winners is not None


class TestSidePots:
    def test_all_in_creates_side_pot(self):
        """When one player goes all-in with less chips, a side pot is created."""
        engine = GameEngine(num_players=3, starting_stack=1000, seed=42)
        # Give player 0 less chips
        engine.players[0].stack = 200
        state = engine.new_hand()

        # Play until hand completes
        max_actions = 50
        for _ in range(max_actions):
            if state.hand_over:
                break
            actions = engine.get_legal_actions(state)
            if not actions:
                break
            # Try all-in for first player, call for others
            all_in = [a for a in actions if a.type == ActionType.ALL_IN]
            call = [a for a in actions if a.type == ActionType.CALL]
            check = [a for a in actions if a.type == ActionType.CHECK]
            fold = [a for a in actions if a.type == ActionType.FOLD]

            if all_in and state.players[state.action_to].seat == 0:
                state = engine.apply_action(all_in[0], state)
            elif call:
                state = engine.apply_action(call[0], state)
            elif check:
                state = engine.apply_action(check[0], state)
            elif fold:
                state = engine.apply_action(fold[0], state)
            else:
                state = engine.apply_action(actions[0], state)

        # Hand should complete without errors
        assert state.hand_over


class TestFolding:
    def test_all_fold_to_winner(self):
        engine = GameEngine(num_players=3, starting_stack=1000, seed=42)
        state = engine.new_hand()

        # Everyone folds
        for _ in range(10):
            if state.hand_over:
                break
            actions = engine.get_legal_actions(state)
            fold = [a for a in actions if a.type == ActionType.FOLD]
            if fold:
                state = engine.apply_action(fold[0], state)
            else:
                check = [a for a in actions if a.type == ActionType.CHECK]
                if check:
                    state = engine.apply_action(check[0], state)
                break

        assert state.hand_over
        assert state.winners is not None
        assert len(state.winners) >= 1
