"""Advisor — provides strategy recommendations based on equity and CFR."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poker_bot.ai.abstraction.action_abstraction import ActionAbstraction
from poker_bot.ai.abstraction.card_abstraction import CardAbstraction
from poker_bot.ai.abstraction.isomorphism import preflop_hand_key
from poker_bot.ai.strategy import StrategyStore
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.card import Card
from poker_bot.game.hand_eval import HandRank, best_hand, monte_carlo_equity
from poker_bot.game.state import GameState, Street


class HandStrength:
    """Classification of hand strength."""

    CATEGORIES = [
        "Monster",       # top set+, straight flush, quads
        "Very Strong",   # overpair, top pair top kicker, two pair, set
        "Strong",        # top pair, strong draw
        "Medium",        # middle pair, weak top pair
        "Weak",          # low pair, weak draw
        "Nothing",       # no made hand, no draw
    ]

    @staticmethod
    def classify(hole_cards: list[Card], board: list[Card]) -> str:
        if not board:
            return HandStrength._classify_preflop(hole_cards)
        return HandStrength._classify_postflop(hole_cards, board)

    @staticmethod
    def _classify_preflop(hole_cards: list[Card]) -> str:
        r1, r2 = hole_cards[0].rank.value, hole_cards[1].rank.value
        suited = hole_cards[0].suit == hole_cards[1].suit
        high = max(r1, r2)
        low = min(r1, r2)

        if r1 == r2:
            if r1 >= 12:  # QQ+
                return "Monster"
            if r1 >= 9:  # 99+
                return "Very Strong"
            return "Strong"

        if high == 14:  # Ace
            if low >= 12:  # AK, AQ
                return "Very Strong"
            if low >= 10 or suited:
                return "Strong"
            return "Medium"

        if suited and high - low <= 2 and high >= 10:
            return "Strong"

        if high >= 12 and low >= 10:
            return "Medium"

        return "Weak" if (suited or high >= 10) else "Nothing"

    @staticmethod
    def _classify_postflop(hole_cards: list[Card], board: list[Card]) -> str:
        result = best_hand(hole_cards, board)
        rank = result.hand_rank

        if rank >= HandRank.FOUR_OF_A_KIND:
            return "Monster"
        if rank >= HandRank.FLUSH:
            return "Monster"
        if rank >= HandRank.STRAIGHT:
            return "Very Strong"
        if rank >= HandRank.THREE_OF_A_KIND:
            return "Very Strong"
        if rank >= HandRank.TWO_PAIR:
            return "Strong"
        if rank >= HandRank.PAIR:
            return "Medium"
        return "Nothing"


class Advisor:
    """Provides play recommendations using equity and/or CFR strategy."""

    def __init__(self, strategy_dir: Path | None = None) -> None:
        self.strategy_store: StrategyStore | None = None
        self.card_abs: CardAbstraction | None = None

        if strategy_dir and strategy_dir.exists():
            self.strategy_store = StrategyStore()
            self.strategy_store.load(strategy_dir / "final")
            self.card_abs = CardAbstraction()
            abs_path = strategy_dir / "abstraction"
            if abs_path.exists():
                self.card_abs.load(abs_path)

    @property
    def has_trained_model(self) -> bool:
        return self.strategy_store is not None and self.strategy_store.size > 0

    def get_advice(
        self,
        hole_cards: list[Card],
        board: list[Card],
        state: GameState,
        num_opponents: int = 1,
    ) -> dict:
        """Get comprehensive advice for the current situation."""
        advice = {}

        # Tier 1: Always available
        equity = monte_carlo_equity(
            hole_cards, board,
            num_opponents=num_opponents,
            num_simulations=2000,
        )
        advice["equity"] = equity

        strength = HandStrength.classify(hole_cards, board)
        advice["hand_strength"] = strength

        advice["recommendation"] = self._equity_recommendation(equity, state)

        # Tier 2: With trained model
        if self.has_trained_model:
            cfr_strategy = self._get_cfr_strategy(hole_cards, board, state)
            if cfr_strategy is not None:
                advice["cfr_strategy"] = cfr_strategy
                advice["cfr_recommendation"] = self._format_strategy(cfr_strategy)

        return advice

    def _equity_recommendation(self, equity: float, state: GameState) -> str:
        pot = state.main_pot
        to_call = state.current_bet - state.players[state.action_to].bet_this_street

        if to_call > 0:
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
            if equity > pot_odds + 0.15:
                return "Raise"
            elif equity > pot_odds:
                return "Call"
            else:
                return "Fold"
        else:
            if equity > 0.7:
                return "Bet Large"
            elif equity > 0.5:
                return "Bet"
            elif equity > 0.35:
                return "Check"
            else:
                return "Check"

    def _get_cfr_strategy(
        self, hole_cards: list[Card], board: list[Card], state: GameState
    ) -> np.ndarray | None:
        if not self.strategy_store or not self.card_abs:
            return None

        street = state.street.name.lower()
        bucket = self.card_abs.get_bucket(street, hole_cards, board)

        # Build action history string with "/" street delimiters
        history_str = ""
        for street_idx, street_actions in enumerate(state.action_history):
            if street_idx > 0 and street_actions:
                history_str += "/"
            for seat, action in street_actions:
                idx = ActionAbstraction.action_index(action, state)
                history_str += f"{idx}:"

        key = f"{bucket}:{history_str}"
        return self.strategy_store.lookup(key)

    def _format_strategy(self, strategy: np.ndarray) -> dict[str, float]:
        action_names = [
            "Fold", "Check", "Call",
            "Bet Small", "Bet Medium", "Bet Large",
            "Bet Pot", "Overbet", "All-In",
        ]
        result = {}
        for i, prob in enumerate(strategy):
            if prob > 0.01:
                name = action_names[i] if i < len(action_names) else f"Action {i}"
                result[name] = float(prob)
        return result
