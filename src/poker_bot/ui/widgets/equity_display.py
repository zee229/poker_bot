"""Equity display panel for advisor mode."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Static


class EquityDisplay(Widget):
    """Right panel showing equity and strategy recommendations."""

    DEFAULT_CSS = """
    EquityDisplay {
        dock: right;
        width: 30;
        background: #16213e;
        padding: 1 2;
        border-left: thick #333;
        layout: vertical;
    }

    EquityDisplay .panel-title {
        color: #e94560;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    EquityDisplay .equity-label {
        color: #aaa;
    }

    EquityDisplay .equity-value {
        color: #2ecc71;
        text-style: bold;
    }

    EquityDisplay .strength-label {
        color: #f0c040;
        margin: 1 0;
    }

    EquityDisplay .recommendation {
        color: #fff;
        text-style: bold;
        margin: 1 0;
        padding: 1;
        background: #2a2a4e;
        text-align: center;
    }

    EquityDisplay .strategy-line {
        color: #ccc;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._advice: dict | None = None

    def compose(self) -> ComposeResult:
        yield Label("Advisor", classes="panel-title")

        if not self._advice:
            yield Label("Play a hand to see advice", classes="equity-label")
            return

        equity = self._advice.get("equity", 0)
        yield Label("Equity:", classes="equity-label")
        yield Label(f"  {equity:.1%}", classes="equity-value")

        bar_len = 20
        filled = int(equity * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        yield Label(f"  [{bar}]", classes="equity-value")

        strength = self._advice.get("hand_strength", "Unknown")
        yield Label(f"Hand: {strength}", classes="strength-label")

        rec = self._advice.get("recommendation", "")
        yield Label(f"→ {rec}", classes="recommendation")

        # CFR strategy if available
        cfr_rec = self._advice.get("cfr_recommendation")
        if cfr_rec:
            yield Label("CFR Strategy:", classes="equity-label")
            for action, prob in cfr_rec.items():
                yield Label(f"  {action}: {prob:.0%}", classes="strategy-line")

    def update_advice(self, advice: dict) -> None:
        self._advice = advice
        self.refresh(recompose=True)
