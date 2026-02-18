# Poker Bot

Texas Hold'em No-Limit (6-max) poker bot with CFR/MCCFR solver, opponent modeling, and ReBeL. Includes a terminal UI for playing against bots.

## Features

- **CFR Solver** — Vanilla CFR and Monte Carlo CFR (external sampling) for computing Nash equilibrium strategies
- **Card Abstraction** — EMD-based hand clustering with suit isomorphism for tractable game trees
- **Action Abstraction** — Configurable bet sizing presets (compact/standard/detailed) with action translation
- **Opponent Modeling** — Tracks VPIP/PFR/AF/C-bet stats, classifies player types (nit, TAG, LAG, calling station, maniac), applies exploitative adjustments
- **ReBeL** — Recursive Belief-based Learning with public belief states and optional value network (PyTorch)
- **Deep CFR** — Neural network function approximation for large games (PyTorch)
- **Subgame Solving** — Real-time strategy refinement at decision points
- **Terminal UI** — Play against bots with keyboard controls via Textual

## Quick Start

```bash
# Install
uv sync

# Play against bots
poker-bot play

# Train on toy games
poker-bot train-toy kuhn --iterations 10000
poker-bot train-toy leduc --iterations 1000

# Precompute card abstraction (run before NLHE training)
poker-bot compute-abstraction

# Train NLHE solver
poker-bot train --players 2 --iterations 10000
```

## Keyboard Controls (Play Mode)

| Key | Action |
|-----|--------|
| `f` | Fold |
| `x` | Check / Call |
| `1` | Bet 1/3 pot |
| `2` | Bet 1/2 pot |
| `3` | Bet 3/4 pot |
| `4` | Bet pot |
| `a` | All-in |
| `n` | New hand |
| `Esc` | Back |

## Project Structure

```
src/poker_bot/
  game/           Game engine (cards, deck, hand evaluation, actions, state, rules)
  ai/
    cfr_base.py       Abstract CFR interface
    vanilla_cfr.py    Full tree traversal CFR
    mccfr.py          Monte Carlo CFR (external sampling)
    parallel_mccfr.py Multi-process MCCFR
    infoset.py        Information set storage with regret matching
    strategy.py       Strategy store (sorted uint64 hashes + binary search)
    trainer.py        NLHE training pipeline
    bot.py            Bot player (CFR strategy + equity fallback + opponent adjustments)
    advisor.py        Strategy advisor
    subgame.py        Subgame solving
    opponent_model.py Opponent stat tracking and exploitation
    abstraction/
      card_abstraction.py    EMD clustering into buckets
      action_abstraction.py  Bet sizing presets
      action_translation.py  Map real bets to abstract actions
      isomorphism.py         Suit isomorphism for preflop
    deep_cfr/         Deep CFR with PyTorch
    rebel/            ReBeL (recursive belief-based learning)
  games/            Toy games (Kuhn, Leduc)
  ui/               Textual TUI (screens, widgets)
tests/              210 tests
```

## Training Pipeline

1. **Compute card abstraction** — clusters hands into buckets per street using equity distributions and EMD
2. **MCCFR training** — traverses abstracted game tree, accumulates regrets, updates strategies
3. **Strategy export** — converts info set regrets to average strategies, stores as sorted numpy arrays
4. **Bot play** — looks up strategy by info set key (card bucket + action history), falls back to equity heuristic

## Requirements

- Python 3.12
- Core: `eval7`, `numpy`, `scipy`, `textual`, `click`
- Optional: `numba` (JIT speedup), `torch` (Deep CFR / ReBeL)

## Install Optional Dependencies

```bash
# Numba for JIT-accelerated regret matching
uv sync --extra fast

# PyTorch for Deep CFR and ReBeL
uv sync --extra torch
```
