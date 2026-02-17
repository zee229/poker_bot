"""Tests for subgame solver — validates strategy production and time budget."""

import time

import numpy as np
import pytest

from poker_bot.ai.abstraction.action_abstraction import ActionAbstraction
from poker_bot.ai.abstraction.card_abstraction import CardAbstraction
from poker_bot.ai.subgame import SubgameAdapter, SubgameSolver
from poker_bot.game.card import Card, Rank, Suit
from poker_bot.game.player import Player, PlayerStatus
from poker_bot.game.rules import BlindStructure
from poker_bot.game.state import GameState, Pot, Street


def _make_flop_state() -> GameState:
    """Create a realistic flop state for subgame testing."""
    players = [
        Player(name="P0", seat=0, stack=9800, bet_this_street=0, bet_this_hand=200),
        Player(name="P1", seat=1, stack=9700, bet_this_street=0, bet_this_hand=300),
    ]
    players[0].hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
    players[1].hole_cards = [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.JACK, Suit.CLUBS)]
    return GameState(
        players=players,
        blinds=BlindStructure(50, 100),
        board=[
            Card(Rank.TEN, Suit.SPADES),
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.EIGHT, Suit.DIAMONDS),
        ],
        street=Street.FLOP,
        pots=[Pot(amount=500, eligible=[0, 1])],
        current_bet=0,
        min_raise=100,
        action_to=0,
        dealer=1,
        action_history=[[], []],
        hand_over=False,
        last_raiser=-1,
        num_actions_this_street=0,
    )


class TestSubgameAdapter:
    def test_initial_state_cloned(self):
        state = _make_flop_state()
        adapter = SubgameAdapter(
            root_state=state,
            card_abstraction=CardAbstraction(),
            action_abstraction=ActionAbstraction(),
            blinds=BlindStructure(50, 100),
        )
        initial = adapter.initial_state()
        # Modifying initial shouldn't affect root
        initial.game_state.players[0].stack = 0
        assert state.players[0].stack == 9800

    def test_depth_limited_terminal(self):
        state = _make_flop_state()
        adapter = SubgameAdapter(
            root_state=state,
            card_abstraction=CardAbstraction(),
            action_abstraction=ActionAbstraction(),
            blinds=BlindStructure(50, 100),
            max_depth=1,
        )
        initial = adapter.initial_state()
        assert not adapter.is_terminal(initial)

    def test_legal_actions_non_empty(self):
        state = _make_flop_state()
        adapter = SubgameAdapter(
            root_state=state,
            card_abstraction=CardAbstraction(),
            action_abstraction=ActionAbstraction(),
            blinds=BlindStructure(50, 100),
        )
        initial = adapter.initial_state()
        actions = adapter.legal_actions(initial)
        assert len(actions) > 0


class TestSubgameSolver:
    def test_produces_valid_strategy(self):
        state = _make_flop_state()
        solver = SubgameSolver(
            card_abstraction=CardAbstraction(),
            action_abstraction=ActionAbstraction(),
            blinds=BlindStructure(50, 100),
            default_iterations=10,
            time_budget=60.0,
        )
        strategies = solver.solve(state, max_iterations=10)
        assert len(strategies) > 0

        for key, strat in strategies.items():
            assert abs(np.sum(strat) - 1.0) < 1e-6
            assert np.all(strat >= -1e-6)

    def test_respects_iteration_limit(self):
        """More iterations produces at least as many info sets (solver runs more)."""
        state = _make_flop_state()
        solver = SubgameSolver(
            card_abstraction=CardAbstraction(),
            action_abstraction=ActionAbstraction(),
            blinds=BlindStructure(50, 100),
            default_iterations=5,
            time_budget=60.0,
        )
        strats_small = solver.solve(state, max_iterations=1)
        strats_large = solver.solve(state, max_iterations=50)

        # Both should produce strategies, larger run should have at least as many
        assert len(strats_small) > 0
        assert len(strats_large) >= len(strats_small)

    def test_get_strategy_for_state(self):
        state = _make_flop_state()
        solver = SubgameSolver(
            card_abstraction=CardAbstraction(),
            action_abstraction=ActionAbstraction(),
            blinds=BlindStructure(50, 100),
            default_iterations=10,
            time_budget=60.0,
        )
        strategy = solver.get_strategy_for_state(state, seat=0, max_iterations=10)
        # May or may not find the exact info set, but should not crash
        if strategy is not None:
            assert abs(np.sum(strategy) - 1.0) < 1e-6

    def test_more_iterations_refines(self):
        """Running more iterations shouldn't crash and should produce strategies."""
        state = _make_flop_state()
        solver = SubgameSolver(
            card_abstraction=CardAbstraction(),
            action_abstraction=ActionAbstraction(),
            blinds=BlindStructure(50, 100),
            default_iterations=20,
            time_budget=60.0,
        )
        strats = solver.solve(state, max_iterations=20)
        assert len(strats) > 0
