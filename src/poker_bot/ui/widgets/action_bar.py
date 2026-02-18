"""Action bar — fold/check/call/raise buttons with pot fraction shortcuts."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Label

from poker_bot.game.actions import Action, ActionType
from poker_bot.game.state import GameState


class ActionBar(Widget):
    """Bottom action bar with poker action buttons."""

    DEFAULT_CSS = """
    ActionBar {
        dock: bottom;
        height: 5;
        background: #16213e;
        padding: 1 2;
        layout: horizontal;
        align: center middle;
    }

    ActionBar Button {
        margin: 0 1;
        min-width: 10;
    }

    ActionBar .fold-btn { background: #e94560; }
    ActionBar .check-btn { background: #2ecc71; }
    ActionBar .call-btn { background: #3498db; }
    ActionBar .raise-btn { background: #f39c12; }
    ActionBar .allin-btn { background: #e74c3c; text-style: bold; }

    ActionBar Input {
        width: 15;
        margin: 0 1;
    }

    ActionBar .size-btn {
        background: #555;
        min-width: 6;
    }
    """

    class ActionSelected(Message):
        def __init__(self, action: Action) -> None:
            super().__init__()
            self.action = action

    def __init__(self, state: GameState | None = None, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        if not self._state or self._state.hand_over:
            yield Label("Waiting...")
            return

        player = self._state.players[self._state.action_to]
        to_call = self._state.current_bet - player.bet_this_street
        pot = self._state.main_pot

        if to_call > 0:
            yield Button("[f] Fold", id="btn-fold", classes="fold-btn")
            call_amt = min(to_call, player.stack)
            if call_amt >= player.stack:
                yield Button(f"[x] All-In ${call_amt:,}", id="btn-allin-call", classes="allin-btn")
            else:
                yield Button(f"[x] Call ${call_amt:,}", id="btn-call", classes="call-btn")
        else:
            yield Button("[x] Check", id="btn-check", classes="check-btn")

        # Bet/Raise sizes
        if player.stack > to_call:
            remaining = player.stack - to_call
            size_keys = [("1", "1/3", 0.33), ("2", "1/2", 0.5), ("3", "3/4", 0.75), ("4", "Pot", 1.0)]
            if self._state.current_bet == 0:
                for key, label, frac in size_keys:
                    amount = max(int(pot * frac), self._state.blinds.big_blind)
                    if amount < remaining:
                        btn_id = label.replace("/", "-")
                        yield Button(f"[{key}] {label}", id=f"btn-bet-{btn_id}", classes="size-btn")
            else:
                for key, label, frac in size_keys[1:]:  # skip 1/3 for raises
                    raise_amount = int(pot * frac)
                    raise_to = self._state.current_bet + raise_amount
                    chips_needed = raise_to - player.bet_this_street
                    if 0 < chips_needed < player.stack:
                        btn_id = label.replace("/", "-")
                        yield Button(f"[{key}] R {label}", id=f"btn-raise-{btn_id}", classes="raise-btn")

            yield Button(f"[a] All-In ${player.stack:,}", id="btn-allin", classes="allin-btn")

            yield Input(placeholder="Custom...", id="bet-input", type="integer")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self._state:
            return

        player = self._state.players[self._state.action_to]
        to_call = self._state.current_bet - player.bet_this_street
        pot = self._state.main_pot
        btn_id = event.button.id

        action: Action | None = None

        if btn_id == "btn-fold":
            action = Action.fold()
        elif btn_id == "btn-check":
            action = Action.check()
        elif btn_id == "btn-call":
            action = Action.call(min(to_call, player.stack))
        elif btn_id == "btn-allin-call":
            action = Action.all_in(player.stack)
        elif btn_id == "btn-allin":
            action = Action.all_in(player.stack)
        elif btn_id and btn_id.startswith("btn-bet-"):
            frac_map = {"1-3": 0.33, "1-2": 0.5, "3-4": 0.75, "Pot": 1.0}
            label = btn_id.replace("btn-bet-", "")
            frac = frac_map.get(label, 0.5)
            amount = max(int(pot * frac), self._state.blinds.big_blind)
            action = Action.bet(amount)
        elif btn_id and btn_id.startswith("btn-raise-"):
            frac_map = {"1-2": 0.5, "3-4": 0.75, "Pot": 1.0}
            label = btn_id.replace("btn-raise-", "")
            frac = frac_map.get(label, 0.5)
            raise_amount = int(pot * frac)
            raise_to = self._state.current_bet + raise_amount
            action = Action.raise_to(raise_to)

        if action:
            self.post_message(self.ActionSelected(action))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not self._state:
            return
        try:
            amount = int(event.value)
        except ValueError:
            return

        if self._state.current_bet == 0:
            self.post_message(self.ActionSelected(Action.bet(amount)))
        else:
            self.post_message(self.ActionSelected(Action.raise_to(amount)))

    def update_state(self, state: GameState) -> None:
        self._state = state
        self.refresh(recompose=True)
