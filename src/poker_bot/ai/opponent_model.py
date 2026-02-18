"""Opponent modeling — track stats, classify player types, suggest exploitative adjustments."""

from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from poker_bot.game.actions import Action, ActionType
from poker_bot.game.state import GameState, Street


class PlayerType(enum.Enum):
    UNKNOWN = "unknown"
    NIT = "nit"
    TAG = "tag"
    LAG = "lag"
    CALLING_STATION = "station"
    MANIAC = "maniac"


@dataclass
class OpponentStats:
    """Per-opponent statistics tracked across hands."""

    hands_seen: int = 0
    vpip: int = 0
    pfr: int = 0
    af_bets: int = 0
    af_calls: int = 0
    cbet_opportunities: int = 0
    cbet_made: int = 0
    fold_to_cbet: int = 0
    fold_to_cbet_opportunities: int = 0
    street_bets: dict[Street, int] = field(default_factory=lambda: defaultdict(int))
    street_calls: dict[Street, int] = field(default_factory=lambda: defaultdict(int))

    # Per-hand tracking (reset each hand)
    _hand_vpip: bool = False
    _hand_pfr: bool = False
    _was_preflop_raiser: bool = False

    @property
    def vpip_pct(self) -> float:
        return self.vpip / self.hands_seen if self.hands_seen > 0 else 0.0

    @property
    def pfr_pct(self) -> float:
        return self.pfr / self.hands_seen if self.hands_seen > 0 else 0.0

    @property
    def aggression_factor(self) -> float:
        return self.af_bets / self.af_calls if self.af_calls > 0 else float(self.af_bets)

    @property
    def cbet_pct(self) -> float:
        return self.cbet_made / self.cbet_opportunities if self.cbet_opportunities > 0 else 0.0

    @property
    def fold_to_cbet_pct(self) -> float:
        return self.fold_to_cbet / self.fold_to_cbet_opportunities if self.fold_to_cbet_opportunities > 0 else 0.0

    def start_hand(self) -> None:
        self.hands_seen += 1
        self._hand_vpip = False
        self._hand_pfr = False
        self._was_preflop_raiser = False

    def end_hand(self) -> None:
        if self._hand_vpip:
            self.vpip += 1
        if self._hand_pfr:
            self.pfr += 1


class OpponentModel:
    """Tracks opponent stats, classifies player types, and provides exploitative adjustments."""

    MIN_HANDS_FOR_CLASSIFICATION = 20

    def __init__(self) -> None:
        self._stats: dict[int, OpponentStats] = {}

    def get_stats(self, seat: int) -> OpponentStats:
        if seat not in self._stats:
            self._stats[seat] = OpponentStats()
        return self._stats[seat]

    def start_hand(self, active_seats: list[int]) -> None:
        for seat in active_seats:
            self.get_stats(seat).start_hand()

    def end_hand(self, active_seats: list[int]) -> None:
        for seat in active_seats:
            self.get_stats(seat).end_hand()

    def observe_action(
        self, seat: int, action: Action, state: GameState,
    ) -> None:
        """Record an observed action for the opponent at seat."""
        stats = self.get_stats(seat)
        street = state.street

        if action.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            stats.af_bets += 1
            stats.street_bets[street] += 1
            if street == Street.PREFLOP:
                stats._hand_vpip = True
                stats._hand_pfr = True
                stats._was_preflop_raiser = True
            else:
                stats._hand_vpip = True
                # Check c-bet: flop bet by preflop raiser, first bet on flop
                if street == Street.FLOP and stats._was_preflop_raiser:
                    flop_actions = state.street_actions(Street.FLOP)
                    has_bet = any(a.type in (ActionType.BET, ActionType.RAISE) for _, a in flop_actions)
                    if not has_bet:
                        stats.cbet_opportunities += 1
                        stats.cbet_made += 1

        elif action.type == ActionType.CALL:
            stats.af_calls += 1
            stats.street_calls[street] += 1
            if street == Street.PREFLOP:
                stats._hand_vpip = True
            else:
                stats._hand_vpip = True

        elif action.type == ActionType.FOLD:
            # Check fold to c-bet
            if street == Street.FLOP:
                flop_actions = state.street_actions(Street.FLOP)
                has_bet = any(a.type in (ActionType.BET, ActionType.RAISE) for _, a in flop_actions)
                if has_bet:
                    stats.fold_to_cbet_opportunities += 1
                    stats.fold_to_cbet += 1

        elif action.type == ActionType.CHECK:
            # Check on flop as preflop raiser → missed c-bet opportunity
            if street == Street.FLOP and stats._was_preflop_raiser:
                flop_actions = state.street_actions(Street.FLOP)
                has_bet = any(a.type in (ActionType.BET, ActionType.RAISE) for _, a in flop_actions)
                if not has_bet:
                    stats.cbet_opportunities += 1

    def classify(self, seat: int) -> PlayerType:
        """Classify opponent into a player type based on accumulated stats."""
        stats = self.get_stats(seat)
        if stats.hands_seen < self.MIN_HANDS_FOR_CLASSIFICATION:
            return PlayerType.UNKNOWN

        vpip = stats.vpip_pct
        pfr = stats.pfr_pct
        af = stats.aggression_factor

        # Maniac: very high VPIP + very high aggression
        if vpip > 0.50 and af > 3.0:
            return PlayerType.MANIAC

        # Calling station: high VPIP + low aggression
        if vpip > 0.35 and af < 1.5:
            return PlayerType.CALLING_STATION

        # LAG: moderately high VPIP + high aggression
        if vpip > 0.30 and af >= 2.0:
            return PlayerType.LAG

        # TAG: moderate VPIP + high aggression
        if 0.18 <= vpip <= 0.30 and pfr >= 0.12 and af >= 1.5:
            return PlayerType.TAG

        # Nit: very low VPIP
        if vpip < 0.18:
            return PlayerType.NIT

        return PlayerType.TAG  # default for unclassified

    def get_exploitative_adjustments(self, seat: int) -> dict[str, float]:
        """Return strategy adjustment multipliers based on opponent type.

        Returns dict with keys:
        - 'value_bet_factor': multiply value betting frequency
        - 'bluff_factor': multiply bluffing frequency
        - 'call_factor': multiply calling frequency
        - 'steal_factor': multiply stealing frequency
        """
        player_type = self.classify(seat)

        if player_type == PlayerType.CALLING_STATION:
            return {
                "value_bet_factor": 1.5,
                "bluff_factor": 0.3,
                "call_factor": 0.8,
                "steal_factor": 0.7,
            }
        elif player_type == PlayerType.NIT:
            return {
                "value_bet_factor": 0.8,
                "bluff_factor": 0.5,
                "call_factor": 0.5,
                "steal_factor": 1.8,
            }
        elif player_type == PlayerType.MANIAC:
            return {
                "value_bet_factor": 1.2,
                "bluff_factor": 0.4,
                "call_factor": 1.5,
                "steal_factor": 0.6,
            }
        elif player_type == PlayerType.LAG:
            return {
                "value_bet_factor": 1.1,
                "bluff_factor": 0.7,
                "call_factor": 1.2,
                "steal_factor": 0.8,
            }
        else:
            # TAG or UNKNOWN: play Nash (no adjustment)
            return {
                "value_bet_factor": 1.0,
                "bluff_factor": 1.0,
                "call_factor": 1.0,
                "steal_factor": 1.0,
            }

    def adjust_strategy(
        self, strategy: np.ndarray, seat: int, state: GameState,
    ) -> np.ndarray:
        """Apply exploitative adjustments to a CFR strategy distribution.

        strategy: array of action probabilities indexed by abstract action
        The adjustments skew the distribution based on opponent tendencies.
        """
        player_type = self.classify(seat)
        if player_type in (PlayerType.UNKNOWN, PlayerType.TAG):
            return strategy

        adj = self.get_exploitative_adjustments(seat)
        modified = strategy.copy()

        # Index 0 = fold (bluff-related: if we fold less, we bluff more)
        # Index 1 = check (passive)
        # Index 2 = call
        # Index 3+ = bet/raise sizes (aggressive)
        # Last index = all-in (aggressive)

        n = len(modified)
        if n < 3:
            return strategy

        # Adjust call probability
        if n > 2:
            modified[2] *= adj["call_factor"]

        # Adjust bet/raise probabilities (value bet + bluff combined)
        bet_factor = (adj["value_bet_factor"] + adj["bluff_factor"]) / 2.0
        for i in range(3, n):
            modified[i] *= bet_factor

        # Re-normalize
        total = modified.sum()
        if total > 0:
            modified /= total
        else:
            modified = strategy.copy()

        return modified
