"""Training screen — shows training progress."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, ProgressBar

from poker_bot.ai.mccfr import MCCFR
from poker_bot.games.kuhn import KuhnPoker
from poker_bot.games.leduc import LeducPoker


class TrainingScreen(Screen):
    """Screen for monitoring CFR training."""

    BINDINGS = [
        ("escape", "go_back", "Back to Menu"),
    ]

    DEFAULT_CSS = """
    TrainingScreen {
        align: center middle;
    }

    #training-container {
        width: 60;
        height: auto;
        padding: 2 4;
        background: #16213e;
        border: thick #333;
    }

    .train-title {
        color: #e94560;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    .train-status {
        color: #ccc;
        margin: 1 0;
    }

    .train-result {
        color: #2ecc71;
        margin: 1 0;
    }

    #training-container Button {
        width: 100%;
        margin: 1 0;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._training = False

    def compose(self) -> ComposeResult:
        with Vertical(id="training-container"):
            yield Label("Training", classes="train-title")
            yield Label("Select a game to train:", classes="train-status")
            yield Button("Train Kuhn (fast)", id="btn-kuhn", variant="primary")
            yield Button("Train Leduc (medium)", id="btn-leduc", variant="primary")
            yield Label("", id="status-label", classes="train-status")
            yield Label("", id="result-label", classes="train-result")
            yield ProgressBar(id="progress-bar", total=100, show_eta=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._training:
            return

        if event.button.id == "btn-kuhn":
            self._run_training("kuhn", 50_000)
        elif event.button.id == "btn-leduc":
            self._run_training("leduc", 10_000)

    def _run_training(self, game: str, iterations: int) -> None:
        self._training = True
        status = self.query_one("#status-label", Label)
        result = self.query_one("#result-label", Label)
        progress = self.query_one("#progress-bar", ProgressBar)

        status.update(f"Training {game} for {iterations:,} iterations...")
        result.update("")
        progress.update(progress=0)

        if game == "kuhn":
            g = KuhnPoker()
        else:
            g = LeducPoker()

        cfr = MCCFR(g, seed=42)

        # Run in chunks to update UI
        chunk = max(iterations // 100, 1)
        self._run_chunk(cfr, iterations, chunk, 0, game, status, result, progress)

    def _run_chunk(self, cfr, total, chunk, done, game, status, result, progress):
        remaining = min(chunk, total - done)
        for _ in range(remaining):
            cfr.iterate()
        done += remaining

        pct = done * 100 // total
        progress.update(progress=pct)
        status.update(f"Training {game}: {done:,}/{total:,} iterations ({len(cfr.info_sets)} info sets)")

        if done < total:
            self.set_timer(0.01, lambda: self._run_chunk(cfr, total, chunk, done, game, status, result, progress))
        else:
            exploit = cfr.compute_exploitability()
            result.update(
                f"Done! {len(cfr.info_sets)} info sets, exploitability: {exploit:.6f}"
            )
            self._training = False

    def action_go_back(self) -> None:
        if not self._training:
            self.app.pop_screen()
