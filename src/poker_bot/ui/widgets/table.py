"""Poker table widget — 6-max layout."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static

from poker_bot.game.card import Card
from poker_bot.game.player import Player
from poker_bot.game.state import GameState
from poker_bot.ui.widgets.card_widget import CardWidget
from poker_bot.ui.widgets.player_widget import PlayerWidget


class PokerTable(Widget):
    """6-max poker table layout."""

    DEFAULT_CSS = """
    PokerTable {
        background: #076324;
        border: thick #0a4a1c;
        width: 100%;
        height: 100%;
        padding: 1;
        layout: vertical;
        align: center middle;
    }

    .table-row {
        layout: horizontal;
        align: center middle;
        height: auto;
        width: 100%;
        margin: 1 0;
    }

    .board-container {
        layout: horizontal;
        align: center middle;
        height: 6;
        width: 100%;
        margin: 1 0;
    }

    .pot-label {
        text-align: center;
        color: #f0c040;
        text-style: bold;
        width: 100%;
    }

    .spacer {
        width: 24;
    }
    """

    def __init__(self, state: GameState | None = None, human_seat: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._human_seat = human_seat

    def compose(self) -> ComposeResult:
        if not self._state:
            yield Label("Waiting for hand...", classes="pot-label")
            return

        players = self._state.players
        n = len(players)

        # Top row: seats 1, 2
        with Horizontal(classes="table-row"):
            if n > 1:
                yield self._make_player(1)
            if n > 2:
                yield self._make_player(2)

        # Middle row: seat 0, board, seat 3
        with Horizontal(classes="table-row"):
            yield self._make_player(0)
            with Vertical(classes="board-container"):
                yield Label(f"Pot: ${self._state.main_pot:,}", classes="pot-label")
                with Horizontal():
                    for card in self._state.board:
                        yield CardWidget(card=card, face_up=True)
            if n > 3:
                yield self._make_player(3)

        # Bottom row: seats 5, 4
        with Horizontal(classes="table-row"):
            if n > 5:
                yield self._make_player(5)
            if n > 4:
                yield self._make_player(4)

    def _make_player(self, seat: int) -> PlayerWidget:
        players = self._state.players
        if seat >= len(players):
            return PlayerWidget(
                Player(name="Empty", seat=seat, stack=0),
                show_cards=False,
            )
        p = players[seat]
        show = seat == self._human_seat or (self._state.hand_over and p.is_in_hand)
        is_turn = not self._state.hand_over and self._state.action_to == seat
        pw = PlayerWidget(p, show_cards=show, is_turn=is_turn)
        if is_turn:
            pw.add_class("active")
        return pw

    def update_state(self, state: GameState, human_seat: int = 0) -> None:
        self._state = state
        self._human_seat = human_seat
        self.refresh(recompose=True)
