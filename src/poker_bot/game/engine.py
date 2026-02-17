"""Game engine for NLHE poker — rounds, betting, side pots, showdown."""

from __future__ import annotations

from poker_bot.game.actions import Action, ActionType
from poker_bot.game.card import Card
from poker_bot.game.deck import Deck
from poker_bot.game.hand_eval import best_hand
from poker_bot.game.player import Player, PlayerStatus
from poker_bot.game.rules import BlindStructure, DEFAULT_BLINDS
from poker_bot.game.state import GameState, Pot, Street


class GameEngine:
    def __init__(
        self,
        num_players: int = 6,
        starting_stack: int = 10000,
        blinds: BlindStructure = DEFAULT_BLINDS,
        seed: int | None = None,
    ) -> None:
        self.blinds = blinds
        self.deck = Deck(seed=seed)
        self.players = [
            Player(name=f"Player {i}", seat=i, stack=starting_stack)
            for i in range(num_players)
        ]
        self.dealer = 0
        self.state: GameState | None = None

    def new_hand(self) -> GameState:
        """Start a new hand: post blinds, deal hole cards."""
        self.deck.shuffle()

        for p in self.players:
            p.reset_for_hand()

        sitting = [p for p in self.players if p.status != PlayerStatus.OUT]
        if len(sitting) < 2:
            raise RuntimeError("Need at least 2 players to start a hand")

        state = GameState(
            players=self.players,
            blinds=self.blinds,
            dealer=self.dealer,
            pots=[Pot(amount=0, eligible=[])],
            action_history=[[]],
        )
        self.state = state

        # Post blinds
        sb_seat = self._next_active_seat(self.dealer)
        bb_seat = self._next_active_seat(sb_seat)

        self._post_blind(state, sb_seat, self.blinds.small_blind)
        self._post_blind(state, bb_seat, self.blinds.big_blind)

        # Post antes
        if self.blinds.ante > 0:
            for p in self.players:
                if p.status != PlayerStatus.OUT:
                    self._post_blind(state, p.seat, self.blinds.ante)

        state.current_bet = self.blinds.big_blind
        state.min_raise = self.blinds.big_blind

        # Deal hole cards
        for p in self.players:
            if p.status != PlayerStatus.OUT:
                p.hole_cards = self.deck.deal(2)

        # Action starts left of BB
        state.action_to = self._next_active_seat(bb_seat)
        state.last_raiser = bb_seat

        return state

    def get_legal_actions(self, state: GameState | None = None) -> list[Action]:
        """Return list of legal actions for the current player."""
        state = state or self.state
        if state is None or state.hand_over:
            return []

        player = state.players[state.action_to]
        if not player.is_active:
            return []

        actions: list[Action] = []
        to_call = state.current_bet - player.bet_this_street

        # Fold — always available if there's a bet to face
        if to_call > 0:
            actions.append(Action.fold())

        # Check — only if no bet to call
        if to_call == 0:
            actions.append(Action.check())

        # Call
        if to_call > 0:
            call_amount = min(to_call, player.stack)
            if call_amount < player.stack:
                actions.append(Action.call(call_amount))
            else:
                # Calling puts us all-in
                actions.append(Action.all_in(call_amount))

        # Bet (no previous bet this street)
        if state.current_bet == 0 and player.stack > 0:
            min_bet = state.blinds.big_blind
            if player.stack <= min_bet:
                actions.append(Action.all_in(player.stack))
            else:
                actions.append(Action.bet(min_bet))
                if player.stack > min_bet:
                    actions.append(Action.all_in(player.stack))

        # Raise (there's a bet to face)
        if state.current_bet > 0 and player.stack > to_call:
            min_raise_to = state.current_bet + state.min_raise
            max_raise = player.stack + player.bet_this_street  # total bet if all-in

            if max_raise <= min_raise_to:
                # Can only go all-in (not a full raise)
                actions.append(Action.all_in(player.stack))
            else:
                actions.append(Action.raise_to(min_raise_to))
                if max_raise > min_raise_to:
                    actions.append(Action.all_in(player.stack))

        return actions

    def apply_action(self, action: Action, state: GameState | None = None) -> GameState:
        """Apply an action and return updated game state."""
        state = state or self.state
        if state is None:
            raise RuntimeError("No active hand")
        if state.hand_over:
            raise RuntimeError("Hand is over")

        player = state.players[state.action_to]
        seat = state.action_to

        # Record action
        state.action_history[int(state.street)].append((seat, action))
        state.num_actions_this_street += 1

        match action.type:
            case ActionType.FOLD:
                player.status = PlayerStatus.FOLDED

            case ActionType.CHECK:
                pass

            case ActionType.CALL:
                self._place_bet(state, player, action.amount)

            case ActionType.BET:
                state.min_raise = action.amount
                state.current_bet = action.amount
                state.last_raiser = seat
                self._place_bet(state, player, action.amount)

            case ActionType.RAISE:
                raise_by = action.amount - state.current_bet
                if raise_by >= state.min_raise:
                    state.min_raise = raise_by
                state.current_bet = action.amount
                state.last_raiser = seat
                chips = action.amount - player.bet_this_street
                self._place_bet(state, player, chips)

            case ActionType.ALL_IN:
                total_bet = player.bet_this_street + action.amount
                if total_bet > state.current_bet:
                    raise_by = total_bet - state.current_bet
                    if raise_by >= state.min_raise:
                        state.min_raise = raise_by
                    state.current_bet = total_bet
                    state.last_raiser = seat
                self._place_bet(state, player, action.amount)
                player.status = PlayerStatus.ALL_IN

        # Check if hand is over (only one player in hand)
        if state.num_in_hand <= 1:
            self._end_hand_no_showdown(state)
            return state

        # Advance to next active player or next street
        self._advance(state)

        return state

    def _post_blind(self, state: GameState, seat: int, amount: int) -> None:
        player = state.players[seat]
        actual = min(amount, player.stack)
        self._place_bet(state, player, actual)
        if player.stack == 0:
            player.status = PlayerStatus.ALL_IN

    def _place_bet(self, state: GameState, player: Player, amount: int) -> None:
        actual = min(amount, player.stack)
        player.stack -= actual
        player.bet_this_street += actual
        player.bet_this_hand += actual
        state.pots[0].amount += actual

    def _advance(self, state: GameState) -> None:
        """Advance to next player or next street."""
        next_seat = self._find_next_to_act(state)
        if next_seat is not None:
            state.action_to = next_seat
            return

        # Street is over — check if we need showdown or more streets
        if state.num_active <= 1:
            # Only all-in and/or one active player — deal remaining streets
            self._deal_remaining_and_showdown(state)
            return

        self._next_street(state)

    def _find_next_to_act(self, state: GameState) -> int | None:
        """Find next player who needs to act. Return None if street is complete."""
        num_players = len(state.players)
        seat = state.action_to

        for _ in range(num_players):
            seat = (seat + 1) % num_players
            player = state.players[seat]

            if not player.is_active:
                continue

            # Player needs to act if:
            # 1. They haven't acted yet and there's no bet (everyone gets a turn)
            # 2. They have less bet than the current bet
            # 3. They are the last raiser and everyone has acted
            if seat == state.last_raiser and state.num_actions_this_street > 1:
                return None

            if player.bet_this_street < state.current_bet:
                return seat

            # If current_bet == 0, everyone needs a chance to act
            if state.current_bet == 0 and state.num_actions_this_street < state.num_active:
                return seat

        return None

    def _next_street(self, state: GameState) -> None:
        """Advance to next street: deal community cards, reset bets."""
        # Collect bets into side pots
        self._calculate_pots(state)

        # Reset street state
        for p in state.players:
            p.reset_street_bet()
        state.current_bet = 0
        state.min_raise = self.blinds.big_blind
        state.last_raiser = -1
        state.num_actions_this_street = 0

        if state.street == Street.PREFLOP:
            state.street = Street.FLOP
            state.board.extend(self.deck.deal(3))
        elif state.street == Street.FLOP:
            state.street = Street.TURN
            state.board.extend(self.deck.deal(1))
        elif state.street == Street.TURN:
            state.street = Street.RIVER
            state.board.extend(self.deck.deal(1))
        elif state.street == Street.RIVER:
            self._showdown(state)
            return

        state.action_history.append([])

        # First to act postflop: first active player left of dealer
        state.action_to = self._next_active_seat(state.dealer)

        # Check if only one active player can act (rest all-in)
        if state.num_active <= 1:
            self._deal_remaining_and_showdown(state)

    def _deal_remaining_and_showdown(self, state: GameState) -> None:
        """Deal remaining community cards and go to showdown."""
        self._calculate_pots(state)
        for p in state.players:
            p.reset_street_bet()

        while len(state.board) < 5:
            needed = {Street.PREFLOP: 3, Street.FLOP: 1, Street.TURN: 1}.get(state.street, 0)
            if needed > 0:
                state.board.extend(self.deck.deal(needed))
            state.street = Street(state.street + 1)
            if state.street >= Street.SHOWDOWN:
                break

        self._showdown(state)

    def _showdown(self, state: GameState) -> None:
        """Determine winners and distribute pots."""
        state.street = Street.SHOWDOWN
        state.hand_over = True

        self._calculate_pots(state)

        # Evaluate hands for all players still in hand
        hand_results = {}
        for p in state.players:
            if p.is_in_hand and len(p.hole_cards) == 2:
                hand_results[p.seat] = best_hand(p.hole_cards, state.board)

        # Distribute each pot
        winners: list[tuple[int, int]] = []
        for pot in state.pots:
            if not pot.eligible:
                continue

            eligible_with_hands = [
                s for s in pot.eligible if s in hand_results
            ]
            if not eligible_with_hands:
                continue

            best_score = max(hand_results[s].score for s in eligible_with_hands)
            pot_winners = [s for s in eligible_with_hands if hand_results[s].score == best_score]

            share = pot.amount // len(pot_winners)
            remainder = pot.amount % len(pot_winners)

            for i, seat in enumerate(pot_winners):
                amount = share + (1 if i < remainder else 0)
                state.players[seat].stack += amount
                winners.append((seat, amount))

        state.winners = winners

    def _calculate_pots(self, state: GameState) -> None:
        """Calculate main pot and side pots from current bets."""
        # Gather all bets
        bets: list[tuple[int, int]] = []
        for p in state.players:
            if p.bet_this_street > 0 and p.is_in_hand:
                bets.append((p.seat, p.bet_this_street))

        if not bets:
            # Update eligible for existing pots
            for pot in state.pots:
                pot.eligible = [p.seat for p in state.players if p.is_in_hand]
            return

        # Get all unique bet levels from players who are all-in or in hand
        all_bets = sorted(set(p.bet_this_street for p in state.players if p.is_in_hand and p.bet_this_street > 0))

        # Collect total chips bet this street
        total_new = sum(p.bet_this_street for p in state.players if p.bet_this_street > 0)

        # Existing pot amount (minus new bets)
        existing = state.pots[0].amount - total_new if state.pots else 0
        existing = max(0, existing)

        pots: list[Pot] = []
        prev_level = 0

        for level in all_bets:
            pot_amount = 0
            eligible = []
            for p in state.players:
                contribution = min(p.bet_this_street, level) - min(p.bet_this_street, prev_level)
                if contribution > 0:
                    pot_amount += contribution
                if p.is_in_hand and p.bet_this_street >= level:
                    eligible.append(p.seat)
                elif p.is_in_hand and p.status == PlayerStatus.ALL_IN and p.bet_this_street >= prev_level:
                    # All-in player eligible if they contributed to this level
                    if p.bet_this_street > prev_level:
                        eligible.append(p.seat)

            if pot_amount > 0:
                pots.append(Pot(amount=pot_amount, eligible=eligible))
            prev_level = level

        # Add existing pot to first pot
        if pots and existing > 0:
            pots[0].amount += existing
        elif existing > 0:
            pots.insert(0, Pot(amount=existing, eligible=[p.seat for p in state.players if p.is_in_hand]))

        # Also add eligible all-in players to the pots they qualify for
        for pot in pots:
            for p in state.players:
                if p.is_in_hand and p.seat not in pot.eligible:
                    pot.eligible.append(p.seat)

        if pots:
            state.pots = pots
        else:
            state.pots = [Pot(amount=existing, eligible=[p.seat for p in state.players if p.is_in_hand])]

    def _end_hand_no_showdown(self, state: GameState) -> None:
        """End hand when all but one player has folded."""
        state.hand_over = True
        state.street = Street.SHOWDOWN

        winner = next(p for p in state.players if p.is_in_hand)
        total = sum(p.amount for p in state.pots)
        winner.stack += total
        state.winners = [(winner.seat, total)]

    def _next_active_seat(self, from_seat: int) -> int:
        """Find next active (non-OUT, non-FOLDED) seat clockwise."""
        num = len(self.players)
        seat = from_seat
        for _ in range(num):
            seat = (seat + 1) % num
            if self.players[seat].status not in (PlayerStatus.OUT, PlayerStatus.FOLDED):
                return seat
        raise RuntimeError("No active players found")

    def advance_dealer(self) -> None:
        """Move dealer button to next eligible player."""
        self.dealer = self._next_active_seat(self.dealer)
