"""Action abstraction — fixed pot-fraction bet sizes."""

from __future__ import annotations

from poker_bot.game.actions import Action, ActionType
from poker_bot.game.state import GameState, Street


class ActionAbstraction:
    """Maps continuous bet sizes to a fixed set of pot-fraction actions."""

    # Pot fractions for each street
    PREFLOP_RAISES = [2, 3, 4]  # multiples of BB
    POSTFLOP_FRACTIONS = [0.33, 0.5, 0.75, 1.0, 1.5]

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
            for mult in self.PREFLOP_RAISES:
                raise_to = state.blinds.big_blind * mult
                if raise_to > state.current_bet and raise_to - player.bet_this_street < player.stack:
                    actions.append(Action.raise_to(raise_to))
        else:
            for frac in self.POSTFLOP_FRACTIONS:
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

    @staticmethod
    def action_index(action: Action, state: GameState) -> int:
        """Map an action to an abstract action index."""
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

    @staticmethod
    def num_abstract_actions() -> int:
        """Total number of abstract action buckets."""
        return 9  # fold, check, call, 5 sizes, all-in
