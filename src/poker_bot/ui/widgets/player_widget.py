"""Player seat widget for the poker table."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static

from poker_bot.game.player import Player, PlayerStatus
from poker_bot.ui.widgets.card_widget import CardWidget


class PlayerWidget(Widget):
    """Displays a player's seat with name, stack, bet, and cards."""

    DEFAULT_CSS = """
    PlayerWidget {
        width: 22;
        height: 7;
        border: round #555;
        padding: 0 1;
        margin: 0 1;
        layout: vertical;
    }

    PlayerWidget.active {
        border: round #e94560;
        background: #2a2a4e;
    }

    PlayerWidget.folded {
        opacity: 0.5;
    }

    PlayerWidget .player-name {
        text-style: bold;
        color: #fff;
    }

    PlayerWidget .player-stack {
        color: #f0c040;
    }

    PlayerWidget .player-bet {
        color: #90ee90;
    }

    PlayerWidget .player-cards {
        layout: horizontal;
        height: 1;
    }
    """

    is_turn: reactive[bool] = reactive(False)

    def __init__(self, player: Player, show_cards: bool = False, is_turn: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._player = player
        self._show_cards = show_cards
        self.is_turn = is_turn

    def compose(self) -> ComposeResult:
        p = self._player
        dealer = " [D]" if hasattr(p, "_is_dealer") and p._is_dealer else ""
        yield Label(f"{p.name}{dealer}", classes="player-name")
        yield Label(f"${p.stack:,}", classes="player-stack", id=f"stack-{p.seat}")
        if p.bet_this_street > 0:
            yield Label(f"Bet: ${p.bet_this_street:,}", classes="player-bet")

        if p.hole_cards and p.status != PlayerStatus.FOLDED:
            cards_str = " ".join(c.pretty() for c in p.hole_cards) if self._show_cards else "🂠 🂠"
            yield Label(cards_str, classes="player-cards")

    def update_player(self, player: Player, show_cards: bool = False, is_turn: bool = False) -> None:
        self._player = player
        self._show_cards = show_cards
        self.is_turn = is_turn
        self.remove_class("active", "folded")
        if is_turn:
            self.add_class("active")
        if player.status == PlayerStatus.FOLDED:
            self.add_class("folded")
        self.refresh(recompose=True)
