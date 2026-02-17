"""Deck implementation with shuffle and deal."""

from __future__ import annotations

import random

from poker_bot.game.card import Card, make_deck


class Deck:
    def __init__(self, seed: int | None = None) -> None:
        self._cards = make_deck()
        self._rng = random.Random(seed)
        self._index = 0

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)
        self._index = 0

    def deal(self, n: int = 1) -> list[Card]:
        if self._index + n > len(self._cards):
            raise RuntimeError("Not enough cards in deck")
        cards = self._cards[self._index : self._index + n]
        self._index += n
        return cards

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    @property
    def remaining(self) -> int:
        return len(self._cards) - self._index

    def reset(self, seed: int | None = None) -> None:
        self._cards = make_deck()
        if seed is not None:
            self._rng = random.Random(seed)
        self._index = 0
        self.shuffle()

    def clone(self) -> Deck:
        """Fast clone — avoids copy.deepcopy overhead."""
        new = Deck.__new__(Deck)
        new._cards = list(self._cards)
        new._rng = random.Random()
        new._rng.setstate(self._rng.getstate())
        new._index = self._index
        return new
