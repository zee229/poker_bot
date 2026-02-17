"""ASCII-art card widget with colored Unicode suits."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from poker_bot.game.card import Card, Suit


class CardWidget(Static):
    """Displays a single card as ASCII art."""

    card: reactive[Card | None] = reactive(None)
    face_up: reactive[bool] = reactive(True)

    def __init__(
        self,
        card: Card | None = None,
        face_up: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.card = card
        self.face_up = face_up

    def render(self) -> str:
        if self.card is None:
            return ""

        if not self.face_up:
            return (
                "┌─────┐\n"
                "│░░░░░│\n"
                "│░░░░░│\n"
                "│░░░░░│\n"
                "└─────┘"
            )

        rank = self.card.rank.char
        suit = self.card.suit.symbol

        return (
            f"┌─────┐\n"
            f"│{rank:<2}   │\n"
            f"│  {suit}  │\n"
            f"│   {rank:>2}│\n"
            f"└─────┘"
        )

    def watch_card(self, card: Card | None) -> None:
        self._update_classes()
        self.refresh()

    def watch_face_up(self, face_up: bool) -> None:
        self._update_classes()
        self.refresh()

    def _update_classes(self) -> None:
        self.remove_class("red", "black", "facedown")
        if not self.face_up:
            self.add_class("card", "facedown")
        elif self.card:
            self.add_class("card")
            if self.card.suit in (Suit.HEARTS, Suit.DIAMONDS):
                self.add_class("red")
            else:
                self.add_class("black")


class CardRow(Widget):
    """Displays multiple cards in a row."""

    DEFAULT_CSS = """
    CardRow {
        layout: horizontal;
        height: 5;
        width: auto;
    }
    """

    def __init__(self, cards: list[Card] | None = None, face_up: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._cards = cards or []
        self._face_up = face_up

    def compose(self) -> ComposeResult:
        for card in self._cards:
            yield CardWidget(card=card, face_up=self._face_up)

    def update_cards(self, cards: list[Card], face_up: bool = True) -> None:
        self._cards = cards
        self._face_up = face_up
        self.refresh(recompose=True)
