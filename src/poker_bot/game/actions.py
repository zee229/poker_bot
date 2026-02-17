"""Action types and action representation for poker."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ActionType(enum.Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass(frozen=True, slots=True)
class Action:
    type: ActionType
    amount: int = 0

    def __str__(self) -> str:
        if self.type in (ActionType.FOLD, ActionType.CHECK):
            return self.type.value
        return f"{self.type.value} {self.amount}"

    @classmethod
    def fold(cls) -> Action:
        return cls(ActionType.FOLD)

    @classmethod
    def check(cls) -> Action:
        return cls(ActionType.CHECK)

    @classmethod
    def call(cls, amount: int) -> Action:
        return cls(ActionType.CALL, amount)

    @classmethod
    def bet(cls, amount: int) -> Action:
        return cls(ActionType.BET, amount)

    @classmethod
    def raise_to(cls, amount: int) -> Action:
        return cls(ActionType.RAISE, amount)

    @classmethod
    def all_in(cls, amount: int) -> Action:
        return cls(ActionType.ALL_IN, amount)
