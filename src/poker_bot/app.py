"""Main Textual application."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from poker_bot.ui.screens.advisor_screen import AdvisorScreen
from poker_bot.ui.screens.main_menu import MainMenuScreen
from poker_bot.ui.screens.play_screen import PlayScreen
from poker_bot.ui.screens.training_screen import TrainingScreen


class PokerApp(App):
    """Poker Bot TUI Application."""

    TITLE = "Poker Bot"
    CSS_PATH = Path(__file__).parent / "ui" / "styles" / "app.tcss"

    SCREENS = {
        "menu": MainMenuScreen,
        "play": PlayScreen,
        "advisor": AdvisorScreen,
        "training": TrainingScreen,
    }

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, mode: str = "menu", **kwargs):
        super().__init__(**kwargs)
        self._mode = mode

    def on_mount(self) -> None:
        if self._mode == "play":
            self.push_screen("play")
        elif self._mode == "advisor":
            self.push_screen("advisor")
        else:
            self.push_screen("menu")
