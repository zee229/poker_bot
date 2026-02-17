"""Tests for fast state cloning — ensures clone independence and correctness."""

import copy

from poker_bot.game.card import Card, Rank, Suit
from poker_bot.game.deck import Deck
from poker_bot.game.player import Player, PlayerStatus
from poker_bot.game.rules import BlindStructure
from poker_bot.game.state import GameState, Pot, Street


class TestPlayerClone:
    def test_clone_independence(self):
        p = Player(name="Alice", seat=0, stack=1000)
        p.hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        clone = p.clone()

        clone.stack = 500
        clone.hole_cards.append(Card(Rank.QUEEN, Suit.DIAMONDS))
        clone.status = PlayerStatus.FOLDED

        assert p.stack == 1000
        assert len(p.hole_cards) == 2
        assert p.status == PlayerStatus.ACTIVE

    def test_clone_preserves_values(self):
        p = Player(name="Bob", seat=3, stack=2000, bet_this_street=100, bet_this_hand=300)
        p.status = PlayerStatus.ALL_IN
        clone = p.clone()

        assert clone.name == "Bob"
        assert clone.seat == 3
        assert clone.stack == 2000
        assert clone.bet_this_street == 100
        assert clone.bet_this_hand == 300
        assert clone.status == PlayerStatus.ALL_IN


class TestDeckClone:
    def test_clone_preserves_deal_index(self):
        deck = Deck(seed=42)
        deck.shuffle()
        deck.deal(5)
        clone = deck.clone()

        assert clone.remaining == deck.remaining
        assert clone._index == deck._index

    def test_clone_independence(self):
        deck = Deck(seed=42)
        deck.shuffle()
        clone = deck.clone()

        original_cards = deck.deal(5)
        clone_cards = clone.deal(5)

        # Same sequence since same RNG state
        assert [str(c) for c in original_cards] == [str(c) for c in clone_cards]

        # But further deals are independent after mutation
        deck.deal(2)
        assert deck.remaining != clone.remaining

    def test_clone_rng_state_preserved(self):
        deck = Deck(seed=42)
        deck.shuffle()
        deck.deal(10)

        clone = deck.clone()
        # Deal from both and verify same cards
        orig = deck.deal(3)
        cloned = clone.deal(3)
        assert [str(c) for c in orig] == [str(c) for c in cloned]


class TestGameStateClone:
    def _make_state(self) -> GameState:
        players = [
            Player(name="P0", seat=0, stack=900, bet_this_street=100, bet_this_hand=100),
            Player(name="P1", seat=1, stack=800, bet_this_street=200, bet_this_hand=200),
        ]
        players[0].hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        players[1].hole_cards = [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.JACK, Suit.CLUBS)]
        return GameState(
            players=players,
            blinds=BlindStructure(50, 100),
            board=[Card(Rank.TEN, Suit.SPADES), Card(Rank.NINE, Suit.HEARTS), Card(Rank.EIGHT, Suit.DIAMONDS)],
            street=Street.FLOP,
            pots=[Pot(amount=300, eligible=[0, 1])],
            current_bet=200,
            min_raise=200,
            action_to=0,
            dealer=1,
            action_history=[[], []],
            hand_over=False,
            last_raiser=1,
            num_actions_this_street=2,
        )

    def test_clone_independence(self):
        state = self._make_state()
        clone = state.clone()

        # Modify clone
        clone.players[0].stack = 0
        clone.board.append(Card(Rank.SEVEN, Suit.CLUBS))
        clone.pots[0].amount = 999
        clone.hand_over = True

        # Original unchanged
        assert state.players[0].stack == 900
        assert len(state.board) == 3
        assert state.pots[0].amount == 300
        assert state.hand_over is False

    def test_clone_preserves_values(self):
        state = self._make_state()
        clone = state.clone()

        assert clone.street == Street.FLOP
        assert clone.current_bet == 200
        assert clone.min_raise == 200
        assert clone.action_to == 0
        assert clone.dealer == 1
        assert clone.last_raiser == 1
        assert clone.num_actions_this_street == 2
        assert len(clone.board) == 3
        assert len(clone.pots) == 1
        assert clone.pots[0].amount == 300

    def test_clone_matches_deepcopy(self):
        """Clone should produce same result as deepcopy for game logic."""
        state = self._make_state()
        clone = state.clone()
        deep = copy.deepcopy(state)

        assert clone.street == deep.street
        assert clone.current_bet == deep.current_bet
        assert clone.min_raise == deep.min_raise
        assert len(clone.players) == len(deep.players)
        for cp, dp in zip(clone.players, deep.players):
            assert cp.stack == dp.stack
            assert cp.seat == dp.seat
            assert cp.status == dp.status

    def test_action_history_independence(self):
        state = self._make_state()
        clone = state.clone()

        from poker_bot.game.actions import Action
        clone.action_history[0].append((0, Action.fold()))

        assert len(state.action_history[0]) == 0
