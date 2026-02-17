"""Card, Rank, Suit representations for poker."""

from __future__ import annotations

import enum
from dataclasses import dataclass
import eval7


class Suit(enum.IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def symbol(self) -> str:
        return _SUIT_SYMBOLS[self]

    @property
    def char(self) -> str:
        return _SUIT_CHARS[self]


_SUIT_SYMBOLS = {
    Suit.CLUBS: "♣",
    Suit.DIAMONDS: "♦",
    Suit.HEARTS: "♥",
    Suit.SPADES: "♠",
}

_SUIT_CHARS = {
    Suit.CLUBS: "c",
    Suit.DIAMONDS: "d",
    Suit.HEARTS: "h",
    Suit.SPADES: "s",
}

_CHAR_TO_SUIT = {v: k for k, v in _SUIT_CHARS.items()}


class Rank(enum.IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def char(self) -> str:
        return _RANK_CHARS[self]


_RANK_CHARS = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}

_CHAR_TO_RANK = {v: k for k, v in _RANK_CHARS.items()}


@dataclass(frozen=True, slots=True)
class Card:
    rank: Rank
    suit: Suit

    @classmethod
    def from_str(cls, s: str) -> Card:
        """Parse card from string like 'As', 'Kh', 'Td', '2c'."""
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s!r}")
        rank = _CHAR_TO_RANK.get(s[0])
        suit = _CHAR_TO_SUIT.get(s[1])
        if rank is None:
            raise ValueError(f"Invalid rank char: {s[0]!r}")
        if suit is None:
            raise ValueError(f"Invalid suit char: {s[1]!r}")
        return cls(rank=rank, suit=suit)

    @property
    def eval7_card(self) -> eval7.Card:
        return eval7.Card(str(self))

    @property
    def int_value(self) -> int:
        """Integer encoding: rank * 4 + suit. Range 8..59."""
        return self.rank.value * 4 + self.suit.value

    def __str__(self) -> str:
        return f"{self.rank.char}{self.suit.char}"

    def __repr__(self) -> str:
        return f"Card({self})"

    def pretty(self) -> str:
        """Return card with unicode suit symbol."""
        return f"{self.rank.char}{self.suit.symbol}"


def make_deck() -> list[Card]:
    """Return a standard 52-card deck."""
    return [Card(rank=r, suit=s) for r in Rank for s in Suit]
