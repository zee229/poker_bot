"""Play screen — human plays against bots."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label

from poker_bot.ai.bot import BotPlayer
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
        ("escape", "go_back", "Back to Menu"),
        ("n", "new_hand", "New Hand"),
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
        strategy_dir = Path("data/nlhe")
        self._bot = BotPlayer(strategy_dir if strategy_dir.exists() else None)

    def compose(self) -> ComposeResult:
        yield HandHistory(id="hand-history")
        with Vertical(id="play-main"):
            yield PokerTable(state=self._state, human_seat=self._human_seat, id="poker-table")
            yield ActionBar(state=self._state, id="action-bar")

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
