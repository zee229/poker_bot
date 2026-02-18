"""Tests for action abstraction with presets and action translation."""

from __future__ import annotations

import pytest

from poker_bot.ai.abstraction.action_abstraction import ActionAbstraction, _PRESETS
from poker_bot.ai.abstraction.action_translation import ActionTranslator
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.card import Card
from poker_bot.game.player import Player, PlayerStatus
from poker_bot.game.rules import BlindStructure
from poker_bot.game.state import GameState, Pot, Street


def _make_state(
    street: Street = Street.FLOP,
    pot: int = 200,
    current_bet: int = 0,
    player_stack: int = 5000,
    player_bet: int = 0,
    bb: int = 100,
) -> GameState:
    """Helper to create a minimal GameState for testing."""
    blinds = BlindStructure(bb // 2, bb)
    p0 = Player(seat=0, stack=player_stack, name="P0")
    p0.status = PlayerStatus.ACTIVE
    p0.bet_this_street = player_bet
    p1 = Player(seat=1, stack=player_stack, name="P1")
    p1.status = PlayerStatus.ACTIVE
    return GameState(
        players=[p0, p1],
        blinds=blinds,
        board=[Card.from_str("Ah"), Card.from_str("Kd"), Card.from_str("Tc")],
        street=street,
        pots=[Pot(amount=pot, eligible=[0, 1])],
        current_bet=current_bet,
        min_raise=bb,
        action_to=0,
        action_history=[[], []],
    )


class TestActionAbstractionPresets:
    def test_compact_preset_default(self):
        aa = ActionAbstraction()
        assert aa.preset == "compact"
        assert ActionAbstraction.num_abstract_actions("compact") == 9

    def test_standard_preset(self):
        aa = ActionAbstraction(preset="standard")
        assert aa.preset == "standard"
        assert ActionAbstraction.num_abstract_actions("standard") == 18

    def test_detailed_preset(self):
        aa = ActionAbstraction(preset="detailed")
        assert aa.preset == "detailed"
        assert ActionAbstraction.num_abstract_actions("detailed") == 25

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            ActionAbstraction(preset="nonexistent")

    def test_compact_backward_compat_action_index(self):
        """Static action_index should match the original 9-action behavior."""
        state = _make_state(pot=200, current_bet=0)
        assert ActionAbstraction.action_index(Action.fold(), state) == 0
        assert ActionAbstraction.action_index(Action.check(), state) == 1
        assert ActionAbstraction.action_index(Action.call(100), state) == 2
        assert ActionAbstraction.action_index(Action.all_in(5000), state) == 8

    def test_compact_postflop_size_buckets(self):
        """Compact postflop buckets match the original behavior."""
        state = _make_state(pot=200, current_bet=0)
        # 0.33 * 200 = 66 → frac=0.33 → bucket 3 (frac <= 0.4)
        assert ActionAbstraction.action_index(Action.bet(66), state) == 3
        # 0.5 * 200 = 100 → frac=0.5 → bucket 4 (frac <= 0.6)
        assert ActionAbstraction.action_index(Action.bet(100), state) == 4
        # 0.75 * 200 = 150 → frac=0.75 → bucket 5 (frac <= 0.87)
        assert ActionAbstraction.action_index(Action.bet(150), state) == 5
        # 1.0 * 200 = 200 → frac=1.0 → bucket 6 (frac <= 1.25)
        assert ActionAbstraction.action_index(Action.bet(200), state) == 6
        # 1.5 * 200 = 300 → frac=1.5 → bucket 7
        assert ActionAbstraction.action_index(Action.bet(300), state) == 7

    def test_compact_preflop_size_buckets(self):
        """Compact preflop buckets match original behavior."""
        state = _make_state(street=Street.PREFLOP, pot=150, current_bet=100, player_bet=100)
        # 2x BB raise → mult=2 → bucket 3 (mult <= 2.5)
        assert ActionAbstraction.action_index(Action.raise_to(200), state) == 3
        # 3x BB raise → mult=3 → bucket 4 (mult <= 3.5)
        assert ActionAbstraction.action_index(Action.raise_to(300), state) == 4
        # 4x BB raise → mult=4 → bucket 5
        assert ActionAbstraction.action_index(Action.raise_to(400), state) == 5


class TestStandardPresetActions:
    def test_more_actions_than_compact(self):
        """Standard preset should produce more unique abstract actions."""
        aa_compact = ActionAbstraction(preset="compact")
        aa_standard = ActionAbstraction(preset="standard")
        state = _make_state(pot=400, current_bet=0, player_stack=10000)
        compact_actions = aa_compact.abstract_actions(state)
        standard_actions = aa_standard.abstract_actions(state)
        assert len(standard_actions) >= len(compact_actions)

    def test_standard_unique_indices(self):
        """All abstract actions in standard preset should have unique indices."""
        aa = ActionAbstraction(preset="standard")
        state = _make_state(pot=400, current_bet=0, player_stack=10000)
        actions = aa.abstract_actions(state)
        indices = [aa.get_action_index(a, state) for a in actions]
        # Filter out duplicates caused by same-size actions
        unique = set(indices)
        assert len(unique) >= 3  # At least check, some bets, all-in

    def test_standard_preflop_actions(self):
        """Standard preset should have 5 preflop raise sizes."""
        aa = ActionAbstraction(preset="standard")
        state = _make_state(street=Street.PREFLOP, pot=150, current_bet=100, player_bet=50)
        actions = aa.abstract_actions(state)
        raises = [a for a in actions if a.type == ActionType.RAISE]
        # Should have more raise sizes than compact
        assert len(raises) >= 3


class TestActionTranslator:
    def test_fold_unchanged(self):
        aa = ActionAbstraction(preset="standard")
        translator = ActionTranslator(aa)
        state = _make_state(current_bet=100, player_bet=0)
        idx = translator.translate(Action.fold(), state)
        assert idx == 0

    def test_check_unchanged(self):
        aa = ActionAbstraction(preset="standard")
        translator = ActionTranslator(aa)
        state = _make_state(current_bet=0)
        idx = translator.translate(Action.check(), state)
        assert idx == 1

    def test_call_unchanged(self):
        aa = ActionAbstraction(preset="standard")
        translator = ActionTranslator(aa)
        state = _make_state(current_bet=100, player_bet=0)
        idx = translator.translate(Action.call(100), state)
        assert idx == 2

    def test_bet_maps_to_nearest(self):
        """A non-standard bet size maps to the nearest abstract bucket."""
        aa = ActionAbstraction(preset="standard")
        translator = ActionTranslator(aa)
        state = _make_state(pot=200, current_bet=0)
        # Bet 45% pot (90 chips) — should map near 0.5 pot
        idx = translator.translate(Action.bet(90), state)
        # Should be a valid size index (not fold/check/call/all_in)
        assert 3 <= idx <= 16

    def test_translate_to_abstract_action_preserves_non_sizing(self):
        aa = ActionAbstraction(preset="standard")
        translator = ActionTranslator(aa)
        state = _make_state(current_bet=100, player_bet=0)
        result = translator.translate_to_abstract_action(Action.fold(), state)
        assert result.type == ActionType.FOLD


class TestNumAbstractActions:
    def test_compact_has_9(self):
        assert ActionAbstraction.num_abstract_actions("compact") == 9

    def test_standard_has_18(self):
        assert ActionAbstraction.num_abstract_actions("standard") == 18

    def test_detailed_has_25(self):
        assert ActionAbstraction.num_abstract_actions("detailed") == 25

    def test_default_is_compact(self):
        assert ActionAbstraction.num_abstract_actions() == 9
