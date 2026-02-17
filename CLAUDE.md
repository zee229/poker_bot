# Poker Bot

CLI poker bot for Texas Hold'em No-Limit (6-max) with CFR solver.

## Stack
- Python 3.12, uv package manager
- Textual (TUI), eval7 (hand eval), numpy/scipy (numerics), click (CLI)

## Project Structure
- `src/poker_bot/game/` — Game engine (card, deck, hand_eval, actions, player, state, engine, rules)
- `src/poker_bot/ai/` — CFR solver (infoset, cfr_base, vanilla_cfr, mccfr, strategy, trainer, advisor, bot)
- `src/poker_bot/ai/abstraction/` — Card/action abstraction, suit isomorphism
- `src/poker_bot/games/` — Toy games (kuhn, leduc) for CFR validation
- `src/poker_bot/ui/` — Textual TUI (screens, widgets, styles)
- `tests/` — 93 tests covering all components

## Key Commands
- `poker-bot play` — Play against bots (TUI)
- `poker-bot advisor` — Strategy advisor mode (TUI)
- `poker-bot train-toy kuhn|leduc --iterations N` — Train on toy games
- `poker-bot compute-abstraction` — Precompute card abstraction buckets
- `poker-bot train --players N --iterations N` — Train NLHE MCCFR

## Architecture Notes
- Card is frozen dataclass with eval7 conversion
- GameEngine.apply_action() returns updated GameState (used for both UI and CFR tree traversal)
- InfoSet stores cumulative regret and strategy; regret matching normalizes positive regrets
- VanillaCFR: full tree traversal, suitable for Kuhn/Leduc
- MCCFR: external sampling, scales to large games
- Exploitability uses two-pass info-set-constrained best response (not omniscient)
- StrategyStore uses sorted uint64 hashes with binary search for O(log n) lookup
- BotPlayer uses CFR strategy when available, falls back to equity-based heuristic
- Card abstraction uses deterministic FNV-1a hash fallback; turn buckets link to river LUT
- Trainer deep-copies state in apply_action; action history uses "/" street delimiters

## Testing
- `uv run pytest tests/` — Run all 93 tests
- Kuhn CFR converges to Nash equilibrium (exploitability < 0.01)
- Leduc CFR exploitability decreases monotonically
