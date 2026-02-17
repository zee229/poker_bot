"""Training pipeline for NLHE MCCFR."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from poker_bot.ai.abstraction.action_abstraction import ActionAbstraction
from poker_bot.ai.abstraction.card_abstraction import CardAbstraction
from poker_bot.ai.abstraction.isomorphism import preflop_hand_key
from poker_bot.ai.cfr_base import GameAdapter
from poker_bot.ai.mccfr import MCCFR
from poker_bot.ai.strategy import StrategyStore
from poker_bot.game.actions import Action, ActionType
from poker_bot.game.card import Card, make_deck
from poker_bot.game.deck import Deck
from poker_bot.game.engine import GameEngine
from poker_bot.game.player import PlayerStatus
from poker_bot.game.rules import BlindStructure
from poker_bot.game.state import GameState, Street


class NLHEState:
    """Wrapper around GameState for CFR traversal."""

    def __init__(
        self,
        game_state: GameState,
        deck: Deck,
        card_abstraction: CardAbstraction,
        action_abstraction: ActionAbstraction,
        action_history_str: str = "",
    ) -> None:
        self.game_state = game_state
        self.deck = deck
        self.card_abstraction = card_abstraction
        self.action_abstraction = action_abstraction
        self.action_history_str = action_history_str

    @property
    def info_set_key(self) -> str:
        gs = self.game_state
        player = gs.players[gs.action_to]
        street = gs.street.name.lower()
        bucket = self.card_abstraction.get_bucket(
            street, player.hole_cards, gs.board
        )
        return f"{bucket}:{self.action_history_str}"


class NLHEGameAdapter(GameAdapter):
    """Adapts NLHE GameEngine for CFR traversal with abstractions."""

    def __init__(
        self,
        num_players: int = 2,
        starting_stack: int = 10000,
        blinds: BlindStructure | None = None,
        card_abstraction: CardAbstraction | None = None,
        action_abstraction: ActionAbstraction | None = None,
        seed: int = 42,
    ) -> None:
        self._num_players = num_players
        self._starting_stack = starting_stack
        self._blinds = blinds or BlindStructure(50, 100)
        self.card_abs = card_abstraction or CardAbstraction()
        self.action_abs = action_abstraction or ActionAbstraction()
        self._seed = seed
        self._rng = random.Random(seed)

    def initial_state(self):
        return "DEAL"

    def is_terminal(self, state) -> bool:
        if isinstance(state, str):
            return False
        return state.game_state.hand_over

    def terminal_utility(self, state, player: int) -> float:
        gs = state.game_state
        initial_stack = self._starting_stack
        final_stack = gs.players[player].stack
        return float(final_stack - initial_stack)

    def current_player(self, state) -> int:
        if isinstance(state, str):
            return -1
        return state.game_state.action_to

    def num_players(self) -> int:
        return self._num_players

    def info_set_key(self, state, player: int) -> str:
        return state.info_set_key

    def legal_actions(self, state) -> list[int]:
        """Return abstract action indices."""
        actions = self.action_abs.abstract_actions(state.game_state)
        indices = []
        seen = set()
        for a in actions:
            idx = ActionAbstraction.action_index(a, state.game_state)
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return sorted(indices)

    def apply_action(self, state, action_idx: int):
        """Apply abstract action to state."""
        gs = state.game_state.clone()
        deck = state.deck.clone()

        # Map abstract index back to concrete action
        actions = self.action_abs.abstract_actions(gs)
        target = None
        for a in actions:
            if ActionAbstraction.action_index(a, gs) == action_idx:
                target = a
                break
        if target is None:
            target = actions[0] if actions else Action.fold()

        prev_street = gs.street

        # Create engine and apply
        engine = GameEngine.__new__(GameEngine)
        engine.blinds = self._blinds
        engine.deck = deck
        engine.players = gs.players
        engine.state = gs

        new_gs = engine.apply_action(target, gs)

        # Street delimiter: "/" when street changes
        sep = "/" if new_gs.street != prev_street else ""
        return NLHEState(
            game_state=new_gs,
            deck=deck,
            card_abstraction=state.card_abstraction,
            action_abstraction=state.action_abstraction,
            action_history_str=state.action_history_str + f"{action_idx}:{sep}",
        )

    def chance_outcomes(self, state) -> list[tuple]:
        # For MCCFR we only need to sample one outcome
        # Return a single deal with probability 1
        engine = GameEngine(
            num_players=self._num_players,
            starting_stack=self._starting_stack,
            blinds=self._blinds,
            seed=self._rng.randint(0, 2**31),
        )
        gs = engine.new_hand()
        return [(NLHEState(
            game_state=gs,
            deck=engine.deck,
            card_abstraction=self.card_abs,
            action_abstraction=self.action_abs,
        ), 1.0)]


class NLHETrainer:
    """Orchestrates NLHE MCCFR training."""

    def __init__(
        self,
        num_players: int = 2,
        output_dir: str = "data/nlhe",
        checkpoint_interval: int = 10_000,
        starting_stack: int = 10000,
        blinds: BlindStructure | None = None,
        num_workers: int = 1,
    ) -> None:
        self.num_players = num_players
        self.output_dir = Path(output_dir)
        self.checkpoint_interval = checkpoint_interval
        self.num_workers = num_workers

        self.card_abs = CardAbstraction()
        abs_path = self.output_dir / "abstraction"
        if abs_path.exists():
            self.card_abs.load(abs_path)
        self.action_abs = ActionAbstraction()

        self.game_adapter = NLHEGameAdapter(
            num_players=num_players,
            starting_stack=starting_stack,
            blinds=blinds or BlindStructure(50, 100),
            card_abstraction=self.card_abs,
            action_abstraction=self.action_abs,
        )

        if num_workers > 1:
            from poker_bot.ai.parallel_mccfr import ParallelMCCFR
            self._parallel = ParallelMCCFR(
                self.game_adapter, num_workers=num_workers, base_seed=42,
            )
            self.cfr = self._parallel.cfr
        else:
            self._parallel = None
            self.cfr = MCCFR(self.game_adapter, seed=42)

    def train(self, iterations: int) -> None:
        """Run MCCFR training for given iterations."""
        print(f"Starting MCCFR training: {iterations} iterations, "
              f"{self.num_players} players, {self.num_workers} workers")
        start = time.time()

        if self._parallel:
            # Parallel training: run in batches of checkpoint_interval
            remaining = iterations
            done = 0
            while remaining > 0:
                batch = min(remaining, self.checkpoint_interval)
                self._parallel.train(batch)
                done += batch
                remaining -= batch

                elapsed = time.time() - start
                n_info = len(self.cfr.info_sets)
                print(f"\n  Checkpoint {done}: {n_info} info sets, {elapsed:.1f}s elapsed")
                self._save_checkpoint(done)
        else:
            for i in tqdm(range(1, iterations + 1), desc="MCCFR"):
                self.cfr.iterate()

                if i % self.checkpoint_interval == 0:
                    elapsed = time.time() - start
                    n_info = len(self.cfr.info_sets)
                    print(f"\n  Checkpoint {i}: {n_info} info sets, {elapsed:.1f}s elapsed")
                    self._save_checkpoint(i)

        # Save final strategy
        self._save_final()
        elapsed = time.time() - start
        print(f"Training complete: {len(self.cfr.info_sets)} info sets in {elapsed:.1f}s")

    def _save_checkpoint(self, iteration: int) -> None:
        store = StrategyStore(max_actions=ActionAbstraction.num_abstract_actions())
        store.from_info_sets(self.cfr.info_sets)
        store.save(self.output_dir / f"checkpoint_{iteration}")

    def _save_final(self) -> None:
        store = StrategyStore(max_actions=ActionAbstraction.num_abstract_actions())
        store.from_info_sets(self.cfr.info_sets)
        store.save(self.output_dir / "final")
