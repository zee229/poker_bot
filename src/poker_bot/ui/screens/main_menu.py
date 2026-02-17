"""Main menu screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, Static


class MainMenuScreen(Screen):
    """Main menu with mode selection."""

    DEFAULT_CSS = """
    MainMenuScreen {
        align: center middle;
    }

    #menu-container {
        width: 50;
        height: auto;
        padding: 2 4;
        background: #16213e;
        border: thick #e94560;
    }

    #menu-container Label {
        text-align: center;
        width: 100%;
    }

    .menu-title {
        color: #e94560;
        text-style: bold;
        margin-bottom: 1;
    }

    .menu-subtitle {
        color: #aaa;
        margin-bottom: 2;
    }

    #menu-container Button {
        width: 100%;
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Label("♠ ♥ POKER BOT ♦ ♣", classes="menu-title")
            yield Label("CFR Solver for Texas Hold'em NL 6-max", classes="menu-subtitle")
            yield Button("Play vs Bot", id="btn-play", variant="primary")
            yield Button("Strategy Advisor", id="btn-advisor", variant="success")
            yield Button("Train Model", id="btn-train", variant="warning")
            yield Button("Quit", id="btn-quit", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-play":
            self.app.push_screen("play")
        elif event.button.id == "btn-advisor":
            self.app.push_screen("advisor")
        elif event.button.id == "btn-train":
            self.app.push_screen("training")
        elif event.button.id == "btn-quit":
            self.app.exit()
