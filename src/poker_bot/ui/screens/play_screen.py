"""Play screen — human plays against bots."""

from __future__ import annotations

import random
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer

from poker_bot.ai.bot import BotPlayer
from poker_bot.ai.opponent_model import OpponentModel
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.engine import GameEngine
from poker_bot.game.player import PlayerStatus
from poker_bot.game.state import GameState, Street
from poker_bot.ui.widgets.action_bar import ActionBar
from poker_bot.ui.widgets.hand_history import HandHistory
from poker_bot.ui.widgets.table import PokerTable


class PlayScreen(Screen):
    """Screen for playing poker against bots."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("n", "new_hand", "New Hand"),
        ("f", "fold", "Fold"),
        ("x", "check_call", "Check/Call"),
        ("1", "bet_small", "Bet 1/3"),
        ("2", "bet_half", "Bet 1/2"),
        ("3", "bet_three_quarter", "Bet 3/4"),
        ("4", "bet_pot", "Bet Pot"),
        ("a", "all_in", "All-In"),
    ]

    DEFAULT_CSS = """
    PlayScreen {
        layout: horizontal;
    }

    #play-main {
        width: 1fr;
        layout: vertical;
    }
    """

    def __init__(self, num_players: int = 6, human_seat: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._num_players = num_players
        self._human_seat = human_seat
        self._engine = GameEngine(num_players=num_players, seed=random.randint(0, 2**31))
        self._engine.players[human_seat].name = "You"
        self._engine.players[human_seat].is_human = True
        for i, p in enumerate(self._engine.players):
            if i != human_seat:
                p.name = f"Bot {i}"
        self._state: GameState | None = None
        self._opponent_model = OpponentModel()
        strategy_dir = Path("data/nlhe")
        self._bot = BotPlayer(
            strategy_dir if strategy_dir.exists() else None,
            opponent_model=self._opponent_model,
        )

    def compose(self) -> ComposeResult:
        yield HandHistory(id="hand-history")
        with Vertical(id="play-main"):
            yield PokerTable(state=self._state, human_seat=self._human_seat, id="poker-table")
            yield ActionBar(state=self._state, id="action-bar")
            yield Footer()

    def on_mount(self) -> None:
        self._start_new_hand()

    def _start_new_hand(self) -> None:
        # End previous hand stats
        if self._state:
            active = [p.seat for p in self._state.players if p.is_in_hand]
            self._opponent_model.end_hand(active)

        history = self.query_one("#hand-history", HandHistory)
        history.clear_log()

        self._engine.advance_dealer()
        self._state = self._engine.new_hand()

        active = [p.seat for p in self._state.players if p.is_in_hand]
        self._opponent_model.start_hand(active)

        history.log_street(Street.PREFLOP)
        self._update_ui()
        self._try_bot_actions()

    def _update_ui(self) -> None:
        table = self.query_one("#poker-table", PokerTable)
        table.update_state(self._state, self._human_seat)

        action_bar = self.query_one("#action-bar", ActionBar)
        if self._state and not self._state.hand_over and self._state.action_to == self._human_seat:
            action_bar.update_state(self._state)
        else:
            action_bar.update_state(None)

    def _is_my_turn(self) -> bool:
        return (
            self._state is not None
            and not self._state.hand_over
            and self._state.action_to == self._human_seat
        )

    def _submit_action(self, action: Action) -> None:
        """Process a human action (from keyboard or button click)."""
        if not self._is_my_turn():
            return

        player = self._state.players[self._state.action_to]
        self._opponent_model.observe_action(player.seat, action, self._state)
        history = self.query_one("#hand-history", HandHistory)
        history.log_action(player.name, action)

        prev_street = self._state.street
        self._state = self._engine.apply_action(action, self._state)

        if self._state.street != prev_street and not self._state.hand_over:
            board_str = " ".join(c.pretty() for c in self._state.board)
            history.log_street(self._state.street, board_str)

        self._update_ui()

        if not self._state.hand_over:
            self._try_bot_actions()
        else:
            self._show_results()
            self._update_ui()

    def _try_bot_actions(self) -> None:
        """Execute bot actions until it's human's turn or hand is over."""
        if not self._state:
            return

        while not self._state.hand_over:
            if self._state.action_to == self._human_seat:
                break

            # Bot makes a decision
            action = self._bot_decide(self._state)
            player = self._state.players[self._state.action_to]

            # Track opponent actions
            self._opponent_model.observe_action(player.seat, action, self._state)

            history = self.query_one("#hand-history", HandHistory)
            history.log_action(player.name, action)

            prev_street = self._state.street
            self._state = self._engine.apply_action(action, self._state)

            if self._state.street != prev_street and not self._state.hand_over:
                board_str = " ".join(c.pretty() for c in self._state.board)
                history.log_street(self._state.street, board_str)

        self._update_ui()

        if self._state.hand_over:
            self._show_results()

    def _bot_decide(self, state: GameState) -> Action:
        """Bot strategy using CFR model or equity-based heuristic."""
        actions = self._engine.get_legal_actions(state)
        if not actions:
            return Action.fold()
        return self._bot.decide(state, state.action_to, actions)

    def _show_results(self) -> None:
        if not self._state or not self._state.winners:
            return
        history = self.query_one("#hand-history", HandHistory)
        for seat, amount in self._state.winners:
            p = self._state.players[seat]
            history.log_winner(p.name, amount)

    # --- Keyboard action handlers ---

    def action_fold(self) -> None:
        if not self._is_my_turn():
            return
        player = self._state.players[self._state.action_to]
        to_call = self._state.current_bet - player.bet_this_street
        if to_call > 0:
            self._submit_action(Action.fold())

    def action_check_call(self) -> None:
        if not self._is_my_turn():
            return
        player = self._state.players[self._state.action_to]
        to_call = self._state.current_bet - player.bet_this_street
        if to_call == 0:
            self._submit_action(Action.check())
        else:
            call_amt = min(to_call, player.stack)
            if call_amt >= player.stack:
                self._submit_action(Action.all_in(player.stack))
            else:
                self._submit_action(Action.call(call_amt))

    def _bet_or_raise(self, frac: float) -> None:
        if not self._is_my_turn():
            return
        player = self._state.players[self._state.action_to]
        to_call = self._state.current_bet - player.bet_this_street
        pot = self._state.main_pot
        if player.stack <= to_call:
            return
        if self._state.current_bet == 0:
            amount = max(int(pot * frac), self._state.blinds.big_blind)
            remaining = player.stack - to_call
            if amount < remaining:
                self._submit_action(Action.bet(amount))
        else:
            raise_amount = int(pot * frac)
            raise_to = self._state.current_bet + raise_amount
            chips_needed = raise_to - player.bet_this_street
            if 0 < chips_needed < player.stack:
                self._submit_action(Action.raise_to(raise_to))

    def action_bet_small(self) -> None:
        self._bet_or_raise(0.33)

    def action_bet_half(self) -> None:
        self._bet_or_raise(0.5)

    def action_bet_three_quarter(self) -> None:
        self._bet_or_raise(0.75)

    def action_bet_pot(self) -> None:
        self._bet_or_raise(1.0)

    def action_all_in(self) -> None:
        if not self._is_my_turn():
            return
        player = self._state.players[self._state.action_to]
        if player.stack > 0:
            self._submit_action(Action.all_in(player.stack))

    # --- Button click handler (still works too) ---

    def on_action_bar_action_selected(self, event: ActionBar.ActionSelected) -> None:
        self._submit_action(event.action)

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_new_hand(self) -> None:
        self._start_new_hand()
