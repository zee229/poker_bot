"""Hand history log widget."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Label

from poker_bot.game.actions import Action
from poker_bot.game.state import GameState, Street


class HandHistory(Widget):
    """Scrollable hand history log."""

    DEFAULT_CSS = """
    HandHistory {
        dock: left;
        width: 30;
        background: #0f0f23;
        padding: 1;
        border-right: thick #333;
    }

    HandHistory .log-entry {
        color: #ccc;
    }

    HandHistory .log-street {
        color: #e94560;
        text-style: bold;
        margin-top: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._entries: list[str] = []
        self._entry_classes: list[str] = []

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            for entry, cls in zip(self._entries, self._entry_classes):
                yield Label(entry, classes=cls)

    def add_entry(self, text: str, is_street: bool = False) -> None:
        self._entries.append(text)
        self._entry_classes.append("log-street" if is_street else "log-entry")
        self.refresh(recompose=True)

    def log_action(self, player_name: str, action: Action) -> None:
        self.add_entry(f"  {player_name}: {action}")

    def log_street(self, street: Street, board_str: str = "") -> None:
        name = street.name.title()
        if board_str:
            self.add_entry(f"--- {name}: {board_str} ---", is_street=True)
        else:
            self.add_entry(f"--- {name} ---", is_street=True)

    def log_winner(self, player_name: str, amount: int) -> None:
        self.add_entry(f"  >> {player_name} wins ${amount:,}")

    def clear_log(self) -> None:
        self._entries.clear()
        self._entry_classes.clear()
        self.refresh(recompose=True)
