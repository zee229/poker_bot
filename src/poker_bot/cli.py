"""CLI entry point for poker-bot."""

from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """Poker Bot — CFR solver for Texas Hold'em NL 6-max."""
    pass


@cli.command()
def play() -> None:
    """Play poker against the bot."""
    from poker_bot.app import PokerApp
    app = PokerApp(mode="play")
    app.run()


@cli.command()
def advisor() -> None:
    """Get real-time strategy advice."""
    from poker_bot.app import PokerApp
    app = PokerApp(mode="advisor")
    app.run()


@cli.command()
@click.argument("game", type=click.Choice(["kuhn", "leduc"]))
@click.option("--iterations", "-n", default=100_000, help="Number of CFR iterations")
@click.option("--output", "-o", default=None, help="Output directory")
def train_toy(game: str, iterations: int, output: str | None) -> None:
    """Train CFR on a toy game (kuhn/leduc)."""
    if game == "kuhn":
        from poker_bot.games.kuhn import KuhnPoker
        from poker_bot.ai.vanilla_cfr import VanillaCFR

        kuhn = KuhnPoker()
        cfr = VanillaCFR(kuhn)
        click.echo(f"Training Kuhn Poker for {iterations} iterations...")

        for i in range(1, iterations + 1):
            cfr.iterate()
            if i % (iterations // 10) == 0:
                click.echo(f"  Iteration {i}/{iterations} — {len(cfr.info_sets)} info sets")

        click.echo("\nFinal strategy:")
        for key in sorted(cfr.info_sets):
            strategy = cfr.info_sets[key].get_average_strategy()
            click.echo(f"  {key}: {strategy}")

    elif game == "leduc":
        from poker_bot.games.leduc import LeducPoker
        from poker_bot.ai.vanilla_cfr import VanillaCFR

        leduc = LeducPoker()
        cfr = VanillaCFR(leduc)
        click.echo(f"Training Leduc Hold'em for {iterations} iterations...")

        for i in range(1, iterations + 1):
            cfr.iterate()
            if i % (iterations // 10) == 0:
                exploitability = cfr.compute_exploitability()
                click.echo(
                    f"  Iteration {i}/{iterations} — "
                    f"{len(cfr.info_sets)} info sets — "
                    f"exploitability: {exploitability:.6f}"
                )

        click.echo(f"\nDone. {len(cfr.info_sets)} info sets trained.")


@cli.command()
@click.option("--output", "-o", default="data/nlhe/abstraction", help="Output directory")
@click.option("--river-samples", default=10000, help="River sampling count")
@click.option("--turn-samples", default=5000, help="Turn sampling count")
def compute_abstraction(output: str, river_samples: int, turn_samples: int) -> None:
    """Precompute card abstraction buckets."""
    from pathlib import Path
    from poker_bot.ai.abstraction.card_abstraction import CardAbstraction

    abs_ = CardAbstraction()
    click.echo(f"Computing river buckets ({river_samples} samples)...")
    abs_.compute_river_buckets(num_samples=river_samples)
    click.echo(f"Computing turn buckets ({turn_samples} samples)...")
    abs_.compute_turn_buckets(num_samples=turn_samples)
    click.echo("Computing preflop buckets...")
    abs_.compute_preflop_buckets()

    out = Path(output)
    abs_.save(out)
    click.echo(f"Saved abstraction to {out}")


@cli.command()
@click.option("--players", "-p", default=6, help="Number of players (2-6)")
@click.option("--iterations", "-n", default=100_000, help="Number of MCCFR iterations")
@click.option("--output", "-o", default="data/nlhe", help="Output directory")
@click.option("--checkpoint", "-c", default=10_000, help="Checkpoint interval")
def train(players: int, iterations: int, output: str, checkpoint: int) -> None:
    """Train MCCFR for NLHE."""
    from poker_bot.ai.trainer import NLHETrainer

    trainer = NLHETrainer(
        num_players=players,
        output_dir=output,
        checkpoint_interval=checkpoint,
    )
    click.echo(f"Training {players}-player NLHE for {iterations} iterations...")
    trainer.train(iterations)
    click.echo("Done.")
