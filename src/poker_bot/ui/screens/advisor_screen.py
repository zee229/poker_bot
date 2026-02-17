"""Advisor screen — play with real-time strategy advice."""

from __future__ import annotations

import random
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label

from poker_bot.ai.advisor import Advisor
from poker_bot.ai.bot import BotPlayer
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.engine import GameEngine
from poker_bot.game.player import PlayerStatus
from poker_bot.game.state import GameState, Street
from poker_bot.ui.widgets.action_bar import ActionBar
from poker_bot.ui.widgets.equity_display import EquityDisplay
from poker_bot.ui.widgets.hand_history import HandHistory
from poker_bot.ui.widgets.table import PokerTable


class AdvisorScreen(Screen):
    """Screen with poker table and advisor panel."""

    BINDINGS = [
        ("escape", "go_back", "Back to Menu"),
        ("n", "new_hand", "New Hand"),
    ]

    DEFAULT_CSS = """
    AdvisorScreen {
        layout: horizontal;
    }

    #advisor-main {
        width: 1fr;
        layout: vertical;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine = GameEngine(num_players=6, seed=random.randint(0, 2**31))
        self._human_seat = 0
        self._engine.players[0].name = "You"
        self._engine.players[0].is_human = True
        for i in range(1, 6):
            self._engine.players[i].name = f"Bot {i}"
        self._state: GameState | None = None

        strategy_dir = Path("data/nlhe")
        self._advisor = Advisor(strategy_dir if strategy_dir.exists() else None)
        self._bot = BotPlayer(strategy_dir if strategy_dir.exists() else None)

    def compose(self) -> ComposeResult:
        yield HandHistory(id="hand-history")
        with Vertical(id="advisor-main"):
            yield PokerTable(state=self._state, human_seat=self._human_seat, id="poker-table")
            yield ActionBar(state=self._state, id="action-bar")
        yield EquityDisplay(id="equity-display")

    def on_mount(self) -> None:
        self._start_new_hand()

    def _start_new_hand(self) -> None:
        history = self.query_one("#hand-history", HandHistory)
        history.clear_log()

        self._engine.advance_dealer()
        self._state = self._engine.new_hand()

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

        # Update advisor
        self._update_advice()

    def _update_advice(self) -> None:
        equity_panel = self.query_one("#equity-display", EquityDisplay)

        if not self._state or self._state.hand_over:
            return

        player = self._state.players[self._human_seat]
        if not player.hole_cards or not player.is_in_hand:
            return

        num_opponents = sum(
            1 for p in self._state.players
            if p.is_in_hand and p.seat != self._human_seat
        )

        advice = self._advisor.get_advice(
            player.hole_cards,
            self._state.board,
            self._state,
            num_opponents=num_opponents,
        )
        equity_panel.update_advice(advice)

    def _try_bot_actions(self) -> None:
        if not self._state:
            return

        while not self._state.hand_over:
            if self._state.action_to == self._human_seat:
                break

            action = self._bot_decide(self._state)
            player = self._state.players[self._state.action_to]

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

    def on_action_bar_action_selected(self, event: ActionBar.ActionSelected) -> None:
        if not self._state or self._state.hand_over:
            return

        action = event.action
        player = self._state.players[self._state.action_to]
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

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_new_hand(self) -> None:
        self._start_new_hand()
