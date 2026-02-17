"""Parallel MCCFR using multiprocessing for independent traversals."""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np

from poker_bot.ai.cfr_base import CFRVariant, GameAdapter
from poker_bot.ai.mccfr import MCCFR

if TYPE_CHECKING:
    from poker_bot.ai.infoset import InfoSetStore


def _worker_run(
    game: GameAdapter,
    iterations: int,
    seed: int,
    variant: CFRVariant,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Run MCCFR iterations in a worker process.

    Returns dict mapping info set key to (cumulative_regret, cumulative_strategy, num_actions).
    """
    cfr = MCCFR(game, seed=seed, variant=variant)
    for _ in range(iterations):
        cfr.iterate()

    result = {}
    for key, info_set in cfr.info_sets.items():
        result[key] = (
            info_set.cumulative_regret.copy(),
            info_set.cumulative_strategy.copy(),
            info_set.num_actions,
        )
    return result


def merge_info_sets(
    target: InfoSetStore,
    worker_results: list[dict[str, tuple[np.ndarray, np.ndarray, int]]],
) -> None:
    """Merge worker results into a target InfoSetStore by summing regrets and strategies."""
    for result in worker_results:
        for key, (regret, strategy, num_actions) in result.items():
            info_set = target.get_or_create(key, num_actions)
            info_set.cumulative_regret += regret
            info_set.cumulative_strategy += strategy


class ParallelMCCFR:
    """Parallel MCCFR training orchestrator.

    Spawns multiple worker processes, each running independent MCCFR iterations,
    then merges results additively.
    """

    def __init__(
        self,
        game: GameAdapter,
        num_workers: int = 4,
        base_seed: int = 42,
        variant: CFRVariant = CFRVariant.VANILLA,
    ) -> None:
        self.game = game
        self.num_workers = num_workers
        self.base_seed = base_seed
        self.variant = variant
        # Main MCCFR instance holds the merged info sets
        self.cfr = MCCFR(game, seed=base_seed, variant=variant)
        self.total_iterations = 0

    @property
    def info_sets(self) -> InfoSetStore:
        return self.cfr.info_sets

    @property
    def iterations(self) -> int:
        return self.total_iterations

    def train(self, iterations: int) -> None:
        """Run parallel MCCFR training.

        Distributes iterations evenly across workers, then merges results.
        """
        iters_per_worker = iterations // self.num_workers
        remainder = iterations % self.num_workers

        worker_args = []
        for i in range(self.num_workers):
            n = iters_per_worker + (1 if i < remainder else 0)
            seed = self.base_seed + i + self.total_iterations
            worker_args.append((self.game, n, seed, self.variant))

        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.starmap(_worker_run, worker_args)

        merge_info_sets(self.cfr.info_sets, results)
        self.total_iterations += iterations

    def compute_exploitability(self) -> float:
        return self.cfr.compute_exploitability()
