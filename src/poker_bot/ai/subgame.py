"""Subgame solver — localized CFR+ for real-time strategy refinement."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from poker_bot.ai.abstraction.action_abstraction import ActionAbstraction
from poker_bot.ai.abstraction.card_abstraction import CardAbstraction
from poker_bot.ai.cfr_base import CFRVariant, GameAdapter
from poker_bot.ai.infoset import InfoSetStore
from poker_bot.ai.strategy import StrategyStore
from poker_bot.ai.vanilla_cfr import VanillaCFR
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.card import make_deck
from poker_bot.game.deck import Deck
from poker_bot.game.engine import GameEngine
from poker_bot.game.player import PlayerStatus
from poker_bot.game.rules import BlindStructure
from poker_bot.game.state import GameState, Street


class SubgameAdapter(GameAdapter):
    """Adapts a subgame rooted at a specific GameState for CFR traversal.

    Depth-limited: at max_depth streets ahead, uses blueprint EV estimate as
    terminal payoff. If no blueprint available, uses equity-based estimate.
    """

    def __init__(
        self,
        root_state: GameState,
        card_abstraction: CardAbstraction,
        action_abstraction: ActionAbstraction,
        blinds: BlindStructure,
        blueprint: StrategyStore | None = None,
        max_depth: int = 2,
        seed: int = 42,
    ) -> None:
        self._root = root_state
        self._card_abs = card_abstraction
        self._action_abs = action_abstraction
        self._blinds = blinds
        self._blueprint = blueprint
        self._max_depth = max_depth
        self._root_street = int(root_state.street)
        self._seed = seed

    def initial_state(self):
        return _SubgameState(
            game_state=self._root.clone(),
            action_history_str="",
        )

    def is_terminal(self, state) -> bool:
        gs = state.game_state
        if gs.hand_over:
            return True
        # Depth-limited: treat as terminal if we've passed max_depth streets
        street_depth = int(gs.street) - self._root_street
        if street_depth >= self._max_depth:
            return True
        return False

    def terminal_utility(self, state, player: int) -> float:
        gs = state.game_state
        if gs.hand_over and gs.winners:
            # Use actual outcome
            initial = self._root.players[player].stack + self._root.players[player].bet_this_hand
            return float(gs.players[player].stack - initial)

        # Depth-limited leaf: estimate EV from pot equity
        pot = gs.main_pot
        n_in = max(gs.num_in_hand, 1)
        fair_share = pot / n_in
        invested = (
            self._root.players[player].stack + self._root.players[player].bet_this_hand
            - gs.players[player].stack - gs.players[player].bet_this_hand
        )
        # Rough estimate: fair share minus what we've put in since root
        return fair_share - invested

    def current_player(self, state) -> int:
        return state.game_state.action_to

    def num_players(self) -> int:
        return len(self._root.players)

    def info_set_key(self, state, player: int) -> str:
        gs = state.game_state
        p = gs.players[player]
        street = gs.street.name.lower()
        bucket = self._card_abs.get_bucket(street, p.hole_cards, gs.board)
        return f"sg:{bucket}:{state.action_history_str}"

    def legal_actions(self, state) -> list[int]:
        actions = self._action_abs.abstract_actions(state.game_state)
        indices = []
        seen = set()
        for a in actions:
            idx = ActionAbstraction.action_index(a, state.game_state)
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return sorted(indices)

    def _make_deck_for_state(self, gs: GameState) -> Deck:
        """Create a deck with remaining cards (excluding dealt cards) for street transitions."""
        used = set()
        for p in gs.players:
            for c in p.hole_cards:
                used.add(c.int_value)
        for c in gs.board:
            used.add(c.int_value)

        deck = Deck(seed=self._seed)
        remaining = [c for c in make_deck() if c.int_value not in used]
        rng = random.Random(self._seed)
        rng.shuffle(remaining)
        deck._cards = remaining
        deck._index = 0
        return deck

    def apply_action(self, state, action_idx: int):
        gs = state.game_state.clone()

        actions = self._action_abs.abstract_actions(gs)
        target = None
        for a in actions:
            if ActionAbstraction.action_index(a, gs) == action_idx:
                target = a
                break
        if target is None:
            target = actions[0] if actions else Action.fold()

        prev_street = gs.street

        engine = GameEngine.__new__(GameEngine)
        engine.blinds = self._blinds
        engine.deck = self._make_deck_for_state(gs)
        engine.players = gs.players
        engine.state = gs

        new_gs = engine.apply_action(target, gs)

        sep = "/" if new_gs.street != prev_street else ""
        return _SubgameState(
            game_state=new_gs,
            action_history_str=state.action_history_str + f"{action_idx}:{sep}",
        )

    def chance_outcomes(self, state) -> list[tuple]:
        # Subgame solver doesn't traverse chance nodes
        return [(state, 1.0)]


@dataclass
class _SubgameState:
    game_state: GameState
    action_history_str: str


class SubgameSolver:
    """Real-time subgame solver using CFR+ on a depth-limited game tree.

    Given a game state, runs localized CFR+ iterations within a time budget
    to produce a refined strategy for the current decision point.
    """

    def __init__(
        self,
        card_abstraction: CardAbstraction,
        action_abstraction: ActionAbstraction,
        blinds: BlindStructure,
        blueprint: StrategyStore | None = None,
        max_depth: int = 2,
        default_iterations: int = 1000,
        time_budget: float = 5.0,
    ) -> None:
        self._card_abs = card_abstraction
        self._action_abs = action_abstraction
        self._blinds = blinds
        self._blueprint = blueprint
        self._max_depth = max_depth
        self._default_iterations = default_iterations
        self._time_budget = time_budget

    def solve(
        self,
        state: GameState,
        max_iterations: int | None = None,
        time_budget: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Solve subgame and return strategy map (info_set_key -> strategy).

        Runs CFR+ for at most max_iterations or until time_budget seconds elapse.
        Returns the average strategy for all discovered info sets.
        """
        max_iter = max_iterations or self._default_iterations
        budget = time_budget or self._time_budget

        adapter = SubgameAdapter(
            root_state=state,
            card_abstraction=self._card_abs,
            action_abstraction=self._action_abs,
            blinds=self._blinds,
            blueprint=self._blueprint,
            max_depth=self._max_depth,
        )

        cfr = VanillaCFR(adapter, variant=CFRVariant.CFR_PLUS)

        start = time.time()
        for _ in range(max_iter):
            if time.time() - start > budget:
                break
            cfr.iterate()

        # Extract average strategies
        strategies = {}
        for key in cfr.info_sets:
            strategies[key] = cfr.info_sets[key].get_average_strategy()
        return strategies

    def get_strategy_for_state(
        self,
        state: GameState,
        seat: int,
        max_iterations: int | None = None,
        time_budget: float | None = None,
    ) -> np.ndarray | None:
        """Solve and return the strategy for a specific player at the given state.

        Returns the strategy array, or None if the info set wasn't found.
        """
        strategies = self.solve(state, max_iterations, time_budget)

        player = state.players[seat]
        street = state.street.name.lower()
        bucket = self._card_abs.get_bucket(street, player.hole_cards, state.board)
        target_key = f"sg:{bucket}:"

        if target_key in strategies:
            return strategies[target_key]

        # Look for any matching prefix
        for key, strat in strategies.items():
            if key.startswith(f"sg:{bucket}:"):
                return strat

        return None
