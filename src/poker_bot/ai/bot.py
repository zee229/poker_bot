"""Bot player — uses CFR strategy when available, equity-based fallback otherwise."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from poker_bot.ai.abstraction.action_abstraction import ActionAbstraction
from poker_bot.ai.abstraction.card_abstraction import CardAbstraction
from poker_bot.ai.strategy import StrategyStore
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.hand_eval import monte_carlo_equity
from poker_bot.game.state import GameState, Street


class BotPlayer:
    """Bot that uses CFR strategy or equity-based heuristic.

    Decision hierarchy:
    1. CFR blueprint strategy (if available and key found)
    2. Subgame solver (if enabled, refines strategy for current decision)
    3. Equity-based heuristic (always available)
    """

    def __init__(
        self,
        strategy_dir: Path | None = None,
        seed: int = 42,
        use_subgame: bool = False,
        subgame_iterations: int = 500,
        subgame_time_budget: float = 2.0,
    ) -> None:
        self._rng = random.Random(seed)
        self._strategy_store: StrategyStore | None = None
        self._card_abs: CardAbstraction | None = None
        self._action_abs = ActionAbstraction()
        self._subgame_solver = None

        if strategy_dir:
            final_path = strategy_dir / "final"
            if final_path.exists():
                self._strategy_store = StrategyStore()
                self._strategy_store.load(final_path)
                self._card_abs = CardAbstraction()
                abs_path = strategy_dir / "abstraction"
                if abs_path.exists():
                    self._card_abs.load(abs_path)

        if use_subgame and self._card_abs:
            from poker_bot.ai.subgame import SubgameSolver
            from poker_bot.game.rules import BlindStructure
            self._subgame_solver = SubgameSolver(
                card_abstraction=self._card_abs,
                action_abstraction=self._action_abs,
                blinds=BlindStructure(50, 100),
                blueprint=self._strategy_store,
                default_iterations=subgame_iterations,
                time_budget=subgame_time_budget,
            )

    @property
    def has_model(self) -> bool:
        return self._strategy_store is not None and self._strategy_store.size > 0

    def decide(self, state: GameState, seat: int, legal_actions: list[Action]) -> Action:
        """Choose an action. Tries CFR strategy first, then subgame, falls back to equity."""
        if not legal_actions:
            return Action.fold()

        # Try CFR strategy
        if self.has_model:
            action = self._cfr_decide(state, seat, legal_actions)
            if action is not None:
                return action

        # Try subgame solving
        if self._subgame_solver:
            action = self._subgame_decide(state, seat, legal_actions)
            if action is not None:
                return action

        return self._equity_decide(state, seat, legal_actions)

    def _cfr_decide(
        self, state: GameState, seat: int, legal_actions: list[Action]
    ) -> Action | None:
        if not self._strategy_store or not self._card_abs:
            return None

        player = state.players[seat]
        street = state.street.name.lower()
        bucket = self._card_abs.get_bucket(street, player.hole_cards, state.board)

        # Build action history with street delimiters
        history_str = ""
        for street_idx, street_actions in enumerate(state.action_history):
            if street_idx > 0 and street_actions:
                history_str += "/"
            for _, action in street_actions:
                idx = ActionAbstraction.action_index(action, state)
                history_str += f"{idx}:"

        key = f"{bucket}:{history_str}"
        strategy = self._strategy_store.lookup(key)
        if strategy is None:
            return None

        # Sample from strategy distribution
        probs = strategy.copy()
        # Zero out illegal abstract action indices
        legal_indices = set()
        for a in legal_actions:
            legal_indices.add(ActionAbstraction.action_index(a, state))

        for i in range(len(probs)):
            if i not in legal_indices:
                probs[i] = 0.0

        total = probs.sum()
        if total <= 0:
            return None
        probs /= total

        chosen_idx = self._rng.choices(range(len(probs)), weights=probs)[0]

        # Map abstract index to concrete action
        for a in legal_actions:
            if ActionAbstraction.action_index(a, state) == chosen_idx:
                return a

        return legal_actions[0]

    def _subgame_decide(
        self, state: GameState, seat: int, legal_actions: list[Action]
    ) -> Action | None:
        """Use subgame solver to refine strategy for current decision."""
        strategy = self._subgame_solver.get_strategy_for_state(state, seat)
        if strategy is None:
            return None

        # Zero out illegal abstract action indices
        probs = strategy.copy()
        legal_indices = set()
        for a in legal_actions:
            legal_indices.add(ActionAbstraction.action_index(a, state))

        for i in range(len(probs)):
            if i not in legal_indices:
                probs[i] = 0.0

        total = probs.sum()
        if total <= 0:
            return None
        probs /= total

        chosen_idx = self._rng.choices(range(len(probs)), weights=probs)[0]

        for a in legal_actions:
            if ActionAbstraction.action_index(a, state) == chosen_idx:
                return a

        return legal_actions[0]

    def _equity_decide(
        self, state: GameState, seat: int, legal_actions: list[Action]
    ) -> Action:
        player = state.players[seat]

        # Estimate equity with reduced simulations for speed
        try:
            num_opponents = sum(
                1 for p in state.players if p.is_in_hand and p.seat != seat
            )
            equity = monte_carlo_equity(
                player.hole_cards,
                state.board,
                num_opponents=max(num_opponents, 1),
                num_simulations=200,
                seed=self._rng.randint(0, 2**31),
            )
        except (ValueError, Exception):
            # If equity calc fails, play conservatively
            equity = 0.3

        to_call = state.current_bet - player.bet_this_street

        check = [a for a in legal_actions if a.type == ActionType.CHECK]
        call = [a for a in legal_actions if a.type == ActionType.CALL]
        raises = [a for a in legal_actions if a.type in (ActionType.RAISE, ActionType.BET)]
        all_in = [a for a in legal_actions if a.type == ActionType.ALL_IN]

        if to_call > 0:
            pot = state.main_pot
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

            if equity > 0.8 and all_in:
                return all_in[0]
            if equity > pot_odds + 0.15 and raises:
                return raises[0]
            if equity > pot_odds:
                return call[0] if call else legal_actions[0]
            return Action.fold()
        else:
            # No bet to face
            if equity > 0.75 and raises:
                # Strong hand — bet big
                return raises[-1] if len(raises) > 1 else raises[0]
            if equity > 0.55 and raises:
                return raises[0]
            return check[0] if check else legal_actions[0]
