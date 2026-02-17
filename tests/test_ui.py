"""Tests for TUI components."""

import pytest

from poker_bot.game.card import Card
from poker_bot.ui.widgets.card_widget import CardWidget


class TestCardWidget:
    def test_card_render(self):
        card = Card.from_str("As")
        widget = CardWidget(card=card, face_up=True)
        rendered = widget.render()
        assert "A" in rendered
        assert "♠" in rendered

    def test_facedown_render(self):
        card = Card.from_str("As")
        widget = CardWidget(card=card, face_up=False)
        rendered = widget.render()
        assert "░" in rendered
        assert "A" not in rendered

    def test_none_card(self):
        widget = CardWidget(card=None)
        assert widget.render() == ""
