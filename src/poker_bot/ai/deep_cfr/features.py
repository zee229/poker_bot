"""Feature extraction for Deep CFR — converts game state to neural network input."""

from __future__ import annotations

import numpy as np

from poker_bot.game.card import Card, Rank, Suit
from poker_bot.game.state import GameState, Street


# Card encoding: 52 one-hot features per card, or rank + suit embedding
NUM_RANKS = 13
NUM_SUITS = 4
CARD_DIM = NUM_RANKS + NUM_SUITS  # rank one-hot + suit one-hot = 17


def encode_card(card: Card) -> np.ndarray:
    """Encode a card as rank one-hot + suit one-hot (17 dims)."""
    vec = np.zeros(CARD_DIM, dtype=np.float32)
    vec[card.rank] = 1.0
    vec[NUM_RANKS + card.suit] = 1.0
    return vec


def encode_cards(cards: list[Card], max_cards: int) -> np.ndarray:
    """Encode a list of cards, zero-padding to max_cards."""
    result = np.zeros(max_cards * CARD_DIM, dtype=np.float32)
    for i, card in enumerate(cards[:max_cards]):
        result[i * CARD_DIM : (i + 1) * CARD_DIM] = encode_card(card)
    return result


def extract_features(state: GameState, player_seat: int) -> np.ndarray:
    """Extract feature vector from game state for the given player.

    Feature layout:
    - Hole cards: 2 * CARD_DIM = 34
    - Board cards: 5 * CARD_DIM = 85
    - Street one-hot: 4 (preflop, flop, turn, river)
    - Pot ratio: 1 (pot / starting_stacks)
    - Stack ratio: 1 (player stack / pot)
    - To-call ratio: 1 (amount to call / pot)
    - Position: 1 (0 = OOP, 1 = IP)
    - Num active players: 1
    Total: 128 features
    """
    player = state.players[player_seat]

    # Card features
    hole_feats = encode_cards(player.hole_cards, 2)
    board_feats = encode_cards(state.board, 5)

    # Street one-hot
    street_feats = np.zeros(4, dtype=np.float32)
    street_idx = min(int(state.street), 3)
    street_feats[street_idx] = 1.0

    # Numeric features
    total_stacks = sum(p.stack + p.bet_this_hand for p in state.players)
    total_stacks = max(total_stacks, 1)
    pot = max(state.main_pot, 1)

    pot_ratio = pot / total_stacks
    stack_ratio = player.stack / pot
    to_call = max(state.current_bet - player.bet_this_street, 0)
    to_call_ratio = to_call / pot
    is_ip = 1.0 if player_seat == state.dealer else 0.0
    num_active = state.num_active / len(state.players)

    numeric = np.array(
        [pot_ratio, stack_ratio, to_call_ratio, is_ip, num_active],
        dtype=np.float32,
    )

    return np.concatenate([hole_feats, board_feats, street_feats, numeric])


FEATURE_DIM = 2 * CARD_DIM + 5 * CARD_DIM + 4 + 5  # = 128
