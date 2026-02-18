"""Action translation — maps real opponent bets to closest abstract action."""

from __future__ import annotations

from poker_bot.ai.abstraction.action_abstraction import ActionAbstraction, _PRESETS
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.state import GameState, Street


class ActionTranslator:
    """Maps real opponent bets to closest abstract action for strategy lookup."""

    def __init__(self, action_abstraction: ActionAbstraction) -> None:
        self._abs = action_abstraction
        self._preset = action_abstraction.preset

    def translate(self, real_action: Action, state: GameState) -> int:
        """Return abstract action index closest to real_action."""
        # Non-sizing actions map directly
        if real_action.type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            return ActionAbstraction.action_index(real_action, state)
        if real_action.type == ActionType.ALL_IN:
            return ActionAbstraction.action_index(real_action, state)

        # For bets/raises: find nearest abstract size by pot fraction / BB multiple
        return self._abs.get_action_index(real_action, state)

    def translate_to_abstract_action(
        self, real_action: Action, state: GameState,
    ) -> Action:
        """Return the closest abstract action (concrete Action object)."""
        if real_action.type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL, ActionType.ALL_IN):
            return real_action

        # Get all abstract actions and find the one whose index matches
        abstract_actions = self._abs.abstract_actions(state)
        target_idx = self.translate(real_action, state)
        for a in abstract_actions:
            if self._abs.get_action_index(a, state) == target_idx:
                return a

        return real_action
