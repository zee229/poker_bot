"""Action abstraction — pot-fraction bet sizes with configurable presets."""

from __future__ import annotations

from poker_bot.game.actions import Action, ActionType
from poker_bot.game.state import GameState, Street

# Preset configurations
_PRESETS = {
    "compact": {
        "preflop": [2, 3, 4],
        "postflop": [0.33, 0.5, 0.75, 1.0, 1.5],
        "num_actions": 9,
    },
    "standard": {
        "preflop": [2.0, 2.5, 3.0, 4.0, 5.0],
        "postflop": [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.33, 1.5, 2.0],
        "num_actions": 18,
    },
    "detailed": {
        "preflop": [2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
        "postflop": [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5],
        "num_actions": 25,
    },
}


class ActionAbstraction:
    """Maps continuous bet sizes to a fixed set of pot-fraction actions."""

    # Keep backward-compat class attrs for compact preset
    PREFLOP_RAISES = _PRESETS["compact"]["preflop"]
    POSTFLOP_FRACTIONS = _PRESETS["compact"]["postflop"]

    def __init__(self, preset: str = "compact") -> None:
        if preset not in _PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(_PRESETS)}")
        cfg = _PRESETS[preset]
        self._preset = preset
        self._preflop_raises: list[float] = cfg["preflop"]
        self._postflop_fractions: list[float] = cfg["postflop"]
        self._num_actions: int = cfg["num_actions"]

    @property
    def preset(self) -> str:
        return self._preset

    def abstract_actions(self, state: GameState) -> list[Action]:
        """Return abstracted set of legal actions."""
        player = state.players[state.action_to]
        actions: list[Action] = []
        to_call = state.current_bet - player.bet_this_street

        # Fold always available if facing a bet
        if to_call > 0:
            actions.append(Action.fold())

        # Check if no bet
        if to_call == 0:
            actions.append(Action.check())

        # Call
        if to_call > 0:
            call_amount = min(to_call, player.stack)
            if call_amount >= player.stack:
                actions.append(Action.all_in(player.stack))
            else:
                actions.append(Action.call(call_amount))

        pot = state.main_pot
        remaining = player.stack - to_call

        if remaining <= 0:
            return actions

        if state.street == Street.PREFLOP:
            for mult in self._preflop_raises:
                raise_to = int(state.blinds.big_blind * mult)
                if raise_to > state.current_bet and raise_to - player.bet_this_street < player.stack:
                    actions.append(Action.raise_to(raise_to))
        else:
            for frac in self._postflop_fractions:
                bet_amount = int(pot * frac)
                if state.current_bet == 0:
                    if bet_amount >= state.blinds.big_blind and bet_amount < player.stack:
                        actions.append(Action.bet(bet_amount))
                else:
                    raise_to = state.current_bet + bet_amount
                    chips_needed = raise_to - player.bet_this_street
                    if chips_needed > 0 and chips_needed < player.stack:
                        actions.append(Action.raise_to(raise_to))

        # All-in always available
        if player.stack > 0:
            actions.append(Action.all_in(player.stack))

        return actions

    def get_action_index(self, action: Action, state: GameState) -> int:
        """Instance method: map action to index for this preset."""
        return _compute_action_index(action, state, self._preset)

    @staticmethod
    def action_index(action: Action, state: GameState) -> int:
        """Map an action to an abstract action index (compact preset, backward compatible)."""
        return _compact_action_index(action, state)

    @staticmethod
    def num_abstract_actions(preset: str = "compact") -> int:
        """Total number of abstract action buckets for a preset."""
        return _PRESETS[preset]["num_actions"]


def _compact_action_index(action: Action, state: GameState) -> int:
    """Original 9-bucket action index (compact preset). Kept for full backward compatibility."""
    if action.type == ActionType.FOLD:
        return 0
    if action.type == ActionType.CHECK:
        return 1
    if action.type == ActionType.CALL:
        return 2

    pot = max(state.main_pot, 1)
    if action.type == ActionType.ALL_IN:
        return 8  # all-in = last index

    # Map bet/raise to pot fraction bucket
    if state.street == Street.PREFLOP:
        mult = action.amount / state.blinds.big_blind
        if mult <= 2.5:
            return 3
        if mult <= 3.5:
            return 4
        return 5
    else:
        amount = action.amount - state.current_bet if action.type == ActionType.RAISE else action.amount
        frac = amount / pot
        if frac <= 0.4:
            return 3
        if frac <= 0.6:
            return 4
        if frac <= 0.87:
            return 5
        if frac <= 1.25:
            return 6
        return 7


def _compute_action_index(action: Action, state: GameState, preset: str) -> int:
    """Map action to index for any preset using nearest-neighbor bucket matching."""
    if preset == "compact":
        return _compact_action_index(action, state)

    if action.type == ActionType.FOLD:
        return 0
    if action.type == ActionType.CHECK:
        return 1
    if action.type == ActionType.CALL:
        return 2

    cfg = _PRESETS[preset]
    num_actions = cfg["num_actions"]

    if action.type == ActionType.ALL_IN:
        return num_actions - 1

    pot = max(state.main_pot, 1)

    if state.street == Street.PREFLOP:
        mult = action.amount / max(state.blinds.big_blind, 1)
        boundaries = cfg["preflop"]
        # Nearest bucket: indices 3..3+len(preflop)-1
        best_idx = 3
        best_dist = abs(mult - boundaries[0])
        for i, b in enumerate(boundaries[1:], 1):
            d = abs(mult - b)
            if d < best_dist:
                best_dist = d
                best_idx = 3 + i
        return min(best_idx, num_actions - 2)
    else:
        amount = action.amount - state.current_bet if action.type == ActionType.RAISE else action.amount
        frac = amount / pot
        boundaries = cfg["postflop"]
        n_preflop = len(cfg["preflop"])
        base = 3 + n_preflop
        best_idx = base
        best_dist = abs(frac - boundaries[0])
        for i, b in enumerate(boundaries[1:], 1):
            d = abs(frac - b)
            if d < best_dist:
                best_dist = d
                best_idx = base + i
        return min(best_idx, num_actions - 2)
