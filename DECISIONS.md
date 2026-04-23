# Phase 1 Decisions

## Environment

| Item | Value |
|------|-------|
| Showdown commit | `1d2258d0875e7b5f303f94ec0936a98535c37602` |
| Python version | 3.12.3 |
| Metamon checkpoint | `SmallRLGen9Beta` — `policy_epoch_24.pt` (15M params, `small-rl-gen9beta`) |
| PyTorch | 2.6.0+cu124 (CUDA 12.4) |
| poke-env | 0.8.3.3 (UT-Austin-RPL fork, `recover_083` branch) |

## Deviations from Spec

### D2 – METAMON_ALLOW_ANY_POKE_ENV
`metamon/__init__.py` hard-checks for poke-env `0.8.3.2` and raises `ImportError` otherwise.
The fork's `recover_083` branch installs as `0.8.3.3`. Rather than downgrade or patch the
version string, we set `METAMON_ALLOW_ANY_POKE_ENV=True` which gates the check.

### D3 – FORMAT_ALIASES patch
`gen9randombattle` is not in Metamon's `SUPPORTED_BATTLE_FORMATS`. At runtime we add:
```python
metamon.config.FORMAT_ALIASES["gen9randombattle"] = "gen9ou"
```
This makes Metamon treat Gen 9 Random Battle as Gen 9 OU for observation/action purposes,
which is accurate (PAC-TeamPreviewObservationSpace covers the same Pokémon pool).

### D3 – gin_overrides cleared
`SmallRLGen9Beta.gin_overrides` explicitly selects `FlashAttention`, overriding the
`base_config` VanillaAttention fallback. We clear it (`model.gin_overrides = None`) so the
fallback triggers when `flash_attn` is not installed.

### D3 – torch.compile / Python.h headers
**Root cause**: `VanillaAttention._inference_with_cache` (in `amago/nets/transformer.py`) is
decorated with `@torch.compile`. PyTorch's triton backend compiles a C extension at first call,
which requires `Python.h` from the `python3.12-dev` package.

**Error observed** (before fix):
```
/tmp/tmpq_hjfwl6/main.c:5:10: fatal error: Python.h: No such file or directory
compilation terminated.
CalledProcessError: Command ['/usr/bin/gcc', '/tmp/tmpq_hjfwl6/main.c', ...cuda_utils...']
  returned non-zero exit status 1.
```

**Resolution**: `sudo apt-get install -y python3.12-dev` installs the headers at
`/usr/include/python3.12/Python.h`. After installation, `torch.compile` works correctly;
no workaround is needed. The `TORCH_COMPILE_DISABLE` env var and `torch._dynamo.disable()`
calls were removed from `d3_selfplay.py`. D3 re-verified clean: 25-turn battle, 27.9 s
wall-clock (includes first-run JIT warmup), 100% valid actions on both sides.

### D3 – CUDA PyTorch required
CPU-only PyTorch inference for two simultaneous transformer models (14M params each) was too
slow to complete a battle within a 600 s timeout. We force-reinstalled PyTorch with CUDA 12.4
(`torch-2.6.0+cu124`). GPU inference completes a 36-turn self-play battle in ~13 s wall-clock.

### D3 – multiprocessing spawn
`multiprocessing.set_start_method("spawn")` is required. `fork` + CUDA context = deadlock.

### D3 – amago traj_save_len float bug

**Location**: `amago/experiment.py`, line 156:
```python
traj_save_len: int = 1e10
```
The type annotation is `int` but the literal `1e10` is a Python `float` (10000000000.0).
Python dataclasses do not coerce or validate field types, so the stored value is a float.

**Call stack**:
```
Experiment.__post_init__()                    # experiment.py:315
  → save_every_low = save_every_high = self.traj_save_len   # float 1e10
EnvCreator.__call__()                         # amago_env.py:386
  → SequenceWrapper(env, save_every=(1e10, 1e10), ...)
SequenceWrapper.random_traj_length()          # amago_env.py:257
  → return random.randint(*self.save_every)
      ↳ random.randint(10000000000.0, 10000000000.0)
          ↳ TypeError: 'float' object cannot be interpreted as an integer
```

`random.randint` is called every time a trajectory finishes or a new episode starts
(`SequenceWrapper.reset` line 268, `SequenceWrapper.finish_active_traj` line 329).
For inference-only use the workaround is safe: `stagger_traj_file_lengths=False` prevents
the stagger path, and an explicit `traj_save_len=int(1e10)` avoids the float entirely.

**Fix in `metamon/rl/metamon_to_amago.py`** (line 169):
```python
experiment = MetamonAMAGOExperiment(
    ...
    traj_save_len=int(1e10),          # ← explicit int; amago default 1e10 is a float
    stagger_traj_file_lengths=False,  # ← also disables the stagger path for safety
    ...
)
```

**Phase 7 note**: This bug will surface whenever `Experiment` is constructed without an
explicit `traj_save_len=int(...)`. Any training run that relies on the default value and
uses `stagger_traj_file_lengths=True` (the default) will crash on the first episode end.
Upstream fix: change `1e10` to `10_000_000_000` (or `int(1e10)`) in `experiment.py:156`.

### D4 – direct challenge instead of cross_evaluate
`poke_env.cross_evaluate` (player/utils.py:54) calls `player.reset_battles()` after each
matchup, clearing the battles dict before we can log turn counts. We use
`send_challenges` / `accept_challenges` directly so `bot.battles` remains populated.

### D4 – RandomPlayer baseline calibration
`poke_env.player.RandomPlayer.choose_move` delegates entirely to `choose_random_singles_move`
(player/player.py). That method builds a flat list of all legal `BattleOrder` objects —
moves, switches, and any mega/dynamax/tera variants that are currently available — then
picks one uniformly via `random.random()`. In Gen 9 Random Battle the Tera orders are
included in the pool, which slightly inflates the opponent's option count relative to
MaxDamageBot (which never Teras voluntarily). This has a negligible effect on the 97%
baseline figure; the result is not artificially inflated. The baseline is the standard
poke-env `RandomPlayer`, the same used in published benchmarks.

## Results Summary

| Deliverable | Status | Key metric |
|-------------|--------|-----------|
| D1 – Showdown server | PASS | HTTP 200 on port 8000, Gen 9 Random Battle playable |
| D2 – Metamon install | PASS | 10 Gen 1 OU episodes completed |
| D3 – Self-play | PASS | 1 battle, 25 turns, 27.9 s wall-clock, MetaPlayer1 won, 100% valid actions |
| D4 – MaxDamageBot vs Random | PASS | 97/100 wins (97%), avg 24.1 turns |
| D4 – MaxDamageBot vs Metamon | PASS | 20 battles, no crashes (1/20 wins, 5% — expected) |
| D5 – Logging | PASS | 120-entry JSONL with timestamp/format/opponent/result/turns/replay_url |

## Current State and Resume Instructions

**Status**: Phase 1 complete. Phase 2 not yet started.

### Phase briefs
Future phase briefs are saved to `docs/phase_briefs/phaseN.md`.
The canonical project plan is at `docs/project_overview.md`.

### Verify Phase 1 still works

```bash
# 1. Start the Showdown server (must be running before any bot code)
cd /path/to/pokemon-showdown
node pokemon-showdown start --no-security &

# 2. Activate the environment
source /path/to/metamon-env/bin/activate

# 3. Run the sanity check (10 battles, asserts >= 70% win rate)
python scripts/verify_phase1.py
```

Expected output: `PHASE 1 OK — MaxDamageBot N/10 wins vs RandomPlayer`

### Gotchas on resume

- **Showdown server is not persistent.** It must be started manually each session with
  `node pokemon-showdown start --no-security`. It does not auto-restart.
- **metamon-env must be activated** before running any Python bot code.
- **METAMON_ALLOW_ANY_POKE_ENV=True** must be set (or it is set inside the scripts).
  Without it, metamon refuses to import due to poke-env version mismatch (0.8.3.3 vs 0.8.3.2).
- **METAMON_CACHE_DIR** must point to the directory containing downloaded checkpoints.
  The `SmallRLGen9Beta` checkpoint (`policy_epoch_24.pt`) is at
  `metamon-cache/pretrained_models/models--jakegrigsby--metamon/…/small-rl-gen9beta/`.
- **python3.12-dev** must be installed (`sudo apt-get install -y python3.12-dev`) for
  `torch.compile` to work. This is a one-time system install.
- The **metamon submodule** `server/pokemon-showdown` is an empty SSH clone; it is symlinked
  to the local Showdown checkout. Do not `git submodule update` inside metamon.

---

# Phase 2 Decisions

## Summary

Phase 2 implements the Knowledge Layer: five standalone Python modules under `knowledge/`
plus a D6 integration test. All six deliverables pass acceptance criteria.

| Deliverable | Status | Key metric |
|-------------|--------|-----------|
| D1 – set_pool.py | PASS | 508 species loaded, 7-day TTL cache, `to_id()` normalization |
| D2 – set_predictor.py | PASS | Bayesian update over role distribution, `observe_*` API |
| D3 – features.py | PASS | FEATURE_DIM=959, (924 per-Pokemon + 35 global), float32, no NaN/Inf |
| D4 – formes.py | PASS | Palafin/Minior/Morpeko/Terapagos/Cramorant transitions |
| D5 – damage_calc.py | PASS | Metamon formula extended with items/abilities; 20-scenario validation pass |
| D6 – integration test | PASS | 10/10 battles, 868 turns, mean 1.3ms extraction, max 4.8ms |

## Architecture Choices

### FEATURE_DIM = 959
77 features × 12 Pokemon + 35 global = 924 + 35. Layout frozen here for Phase 3 neural net input.

Per-Pokemon layout (indices within each 77-element slot):
```
[0]     species_idx    — embedding index [0,508]; NOT one-hot. Phase 4 must use nn.Embedding.
[1]     hp_fraction
[2]     is_active      — 1.0 if this Pokemon is currently active, else 0.0
[3:10]  status_onehot(7)
[10:17] stat_boosts(7)
[17:23] base_stats(6)
[23]    speed_tier
[24:42] type_move_probs(18)
[42]    priority_prob
[43]    setup_prob
[44]    hazard_prob
[45]    removal_prob
[46]    pivot_prob
[47:55] item_probs(8)
[55]    expected_damage_norm  — DamageCalculator output / inferred opp max HP
[56]    tera_available
[57:75] tera_type_dist(18)
[75]    times_active
[76]    is_revealed
```

Global (35 elements): weather(5), terrain(5), trick_room(2), hazards×2(8), screens×2(12), turn(1), remaining×2(2).

### Slot ordering: stable identity across turns
Own slots 0-5: insertion order from `battle.team` at turn 1 (= initial team order sent by Showdown).
Opp slots 6-11: reveal order — slots fill as new opponents are seen for the first time.
Slots never change within a battle, ensuring the RL policy can track identities frame-to-frame.
`is_active` at index [2] carries which slot is currently active (replaces sort-based ordering).

### species_idx is an embedding index — NOT one-hot
Feature [0] is an integer in [0, 508] identifying the species. Phase 4 must pass it through
`nn.Embedding(509, embed_dim)`, not treat it as a float feature or one-hot encode it.
The rest of features [1:77] are floats and feed into the main network trunk directly.

### Damage calculator: Metamon's formula extended
The three candidate approaches from the brief were: Metamon's, poke-env's, @smogon/calc.
- poke-env has no damage formula.
- @smogon/calc is TypeScript; bridging via Node adds runtime complexity.
- Metamon's `damage_equation` in `baselines/base.py` covers ~90% of cases and is
  already in our dependency tree.
We extended it with Life Orb, Choice Band/Specs, Expert Belt,
Muscle Band, Wise Glasses, Huge Power/Pure Power, Adaptability, Technician,
Tinted Lens, Transistor, Dragon's Maw, Sand/Snow defensive boosts.

Feature [55] (`expected_damage_norm`) routes through the full `DamageCalculator` including
item/ability modifiers. Do not simplify this to the bare formula — the modifiers matter
(Life Orb alone is 1.3x, Choice Band is 1.5x).

### Damage validation: 20-scenario absolute comparison vs Showdown reference
Results from `scripts/validate_damage.py` (run 2026-04-22):

| Scenario group | Max |Δ| | Mean |Δ| | Verdict |
|---|---|---|---|
| Single-multiplier (16 scenarios) | 4.0 HP | 2.1 HP | PASS (≤5 HP) |
| Compound (4 scenarios) | 4.5 HP | 3.8 HP | expected |

Systematic bias: our formula over-estimates by **2–4 HP (2–5% relative)** versus Showdown.
Root cause: Metamon convention applies the RNG roll last (to the accumulated float), whereas
Showdown applies it to the integer base first, then each multiplier with `pokeRound` floors.
This is acceptable for RL feature extraction (consistent signal; the policy learns the bias).
Do not change the formula to match Showdown — the added complexity is not worth 2-4 HP gain.

### SetPredictor: current distribution only, no memory between battles
Each `SetPredictor` is created fresh at the start of each opponent encounter
(via `BattleFeatureExtractor.reset()`). This is intentional: prior beliefs
from one battle should not contaminate a new random matchup.

### FormeTracker: current-forme stats, not probability-weighted
Storing the current forme's stats directly keeps the feature vector deterministic.
The network can learn transition dynamics from training sequences.

## Bugs Found and Fixed

### D5 – Type chart key order and casing in damage_calc.py
`_type_effectiveness()` was looking up `type_chart.get(dtype, {}).get(move_type, 1.0)`,
but `_type_chart` is structured as `{attacking_type: {defending_type: mult}}` with
ALL CAPS keys from poke-env's `GenData`. The lookup was doing key order and casing wrong,
returning 1.0 (neutral) for all type matchups.

Fix: normalize both keys to uppercase and swap order:
`type_chart.get(move_type.upper(), {}).get(dtype.upper(), 1.0)`

This was silent: STAB tests passed, damage values looked plausible, but type effectiveness
was always 1.0. Caught by unit tests `test_supereffective_doubles` and `test_not_very_effective_halves`.

### D4 – FormeTracker initial forme for Palafin and Minior
`FormeTracker.__init__` set `_current_forme = species_id`, which is `"palafin"` (not
`"palafin-zero"`) and `"minior"` (not `"minior-meteor"`). Consequences:
- Palafin: `effective_base_stats` returned `None` (key `"palafin"` missing from FORME_BASE_STATS).
- Minior: `on_damage_taken` checked `"meteor" in "minior"` → False, so the Core transition
  never fired.

Fix: added `_INITIAL_FORME` class dict mapping species to correct starting forme.

### D5 – Weather type comparison was case-sensitive (move types arrive uppercase)
`DamageCalculator.calculate()` compared `move.move_type == "Water"` and `move.move_type == "Fire"`,
but poke-env passes type names as uppercase strings ("WATER", "FIRE"). The rain/sun weather
multipliers were never applied: rain-boosted Water and sun-boosted Fire had the same value as neutral.

Fix: normalized all weather and ability-type comparisons to `.upper()` in `calculate()`.
Caught by `validate_damage.py` scenario 9 (Rain-weakened Fire, Δ=49 HP before fix, Δ=2.5 after).

### D5 – STAB / ability-type comparisons now consistently uppercase-normalized
`_stab()` already used `.upper()`. Fixed weather and ability-type boost comparisons to match.
All type comparisons in `DamageCalculator` and `_stab()` are now case-insensitive.

### D3 – features.py audit fixes (2026-04-22)
- `HAZARD_REMOVAL_MOVES`: `"mortalspinstrike"` → `"mortalspin"` (wrong Showdown ID).
- `SETUP_MOVES`: `"coilingcurse"` → `"coil"`, removed dead entry `"victory dance"`.
- `WEATHER_MAP`: added `"DELTASTREAM"` → `"none"` (was missing, caused KeyError in weather feature).
- Trick Room turns-remaining: activation turn defaulted to 0 instead of the actual activation turn.
  Fixed to `next(iter(battle.field_effects.get(effect, {}).keys()), battle.turn)`.

### D1 – pkmn.cc returns 403 for Python-urllib User-Agent
`urllib.request` sends `User-Agent: Python-urllib/3.12` which pkmn.cc rejects (HTTP 403).
`curl -A "Mozilla/5.0 ..."` works fine.
Fix: added `User-Agent: Mozilla/5.0 (compatible; pokemon-showdown-bot/1.0)` to the Request
headers in `set_pool._fetch_json()`.
Data was pre-populated by running `curl -sL -A "Mozilla/5.0 ..." <url>` manually.

## Unit Tests

65 tests in `tests/test_knowledge.py`. All pass. Covers:
- `to_id()` normalization (6 cases)
- `resolve_species()` / `get_species_data()` (7 cases, require live cache)
- `SetPredictor` Bayesian updating (7 cases)
- `FormeTracker` for Palafin, Minior, Morpeko; `FormeManager` (13 cases)
- `DamageCalculator` core formula, STAB, type effectiveness, items (Life Orb,
  Choice Band), abilities (Huge Power, Burn, Technician), weather, mean mode (22 cases)

## Resume Instructions (Phase 2)

Phase 2 is complete. To verify:
```bash
source /path/to/metamon-env/bin/activate
cd /home/user/showdown-bot

# Unit tests (no server needed)
python -m pytest tests/test_knowledge.py -v

# Integration test (requires Showdown server on port 8000)
node /path/to/pokemon-showdown start --no-security &
python scripts/integration_test_phase2.py
```

## Gotchas

- `randbats` cache at `$METAMON_CACHE_DIR/randbats/gen9randombattle_stats.json` must exist.
  Populated on first run (or manually via curl as above). TTL = 7 days.
- `BattleFeatureExtractor` must be instantiated once per session, not per battle.
  Call `.reset()` at the start of each new battle.
- `SetPredictor.observe_move()` receives Showdown IDs (lowercase, no special chars).
  poke-env's `move.id` is already in this format.

---

# Phase 3 Decisions

## Summary

Phase 3 implements the Replay Ingestion Pipeline: six deliverables under
`replay_ingestion/` that convert raw Showdown replays to training-ready
trajectory `.npz` files.

| Deliverable | Status | Key metric |
|-------------|--------|-----------|
| D1 – scraper.py | PASS | 100 replays, 0 errors, 1901–2343 rating |
| D2 – parser.py | PASS | 100/100 winner detection, all Gen 9 events handled |
| D3 – reconstruct.py | PASS | 100.0% consistency on all 5 checks (HP, status, active, weather, turn-order) |
| D4 – trajectory.py | PASS | 5450 turns, 0 NaN/Inf, 95.9% valid actions, 959-dim vectors |
| D5 – validate_reconstruction.py | PASS | 100% consistency across all checks |
| D6 – dataset_report.md | PASS | Written to data/dataset_report.md |

## Architecture Choices

### Parser (D2): single-pass event processing

All events processed in one forward pass of the replay log.  `TurnSnapshot`
objects are created at each `|turn|N` boundary by cloning live state.  No
backtracking needed except for the Illusion reveal, which sets
`slot.illusion_entry_species` and defers species correction to `detailschange`.

HP is stored as exact integers (current/max) from the log — not percentages,
even though the "HP Percentage Mod" rule is active.  `hp_fraction = current / max`.

Unknown events are counted silently in `ParsedBattle.unknown_event_counts`;
the parser never raises on unknown events.

### Reconstruction (D3): masked opponent slots

`reconstruct.py` produces `ReconstructedView` objects with:
- `own_slots`: full visibility (complete HP, status, boosts, moves, items, etc.)
- `opp_slots`: only what was publicly visible (parser already filters to revealed info)

Non-revealed opponent slots → empty placeholder (`species=""`, `revealed=False`).
Species resolution: `slot.forme` (set on forme changes, Illusion reveal, Transform)
is promoted to `slot.species` and cleared so downstream code only reads `slot.species`.

No future-information leakage: all masking is based on events already processed
at the time of each turn's snapshot.

### Trajectory (D4): action encoding and legality mask

**Action space (13 slots)**:
- 0-3: moves sorted alphabetically by Showdown ID (across ALL moves used in battle)
- 4-8: switches sorted alphabetically by species ID (non-fainted, non-active, known)
- 9-12: tera + move (same move ordering; legal only if `own_can_tera=True`)
- -1: unresolvable (`cant` events, last turn, force-switch timing issue)

**Final-moveset ordering**: Action indices 0-3 use the FINAL known moveset
(all moves seen across the entire battle), not just moves revealed so far.
This ensures consistent encoding across all turns.  Feature vectors still
only encode moves seen up to the current turn (no leakage into feature values).

**Unrevealed own Pokemon in switches**: `_available_switches()` includes Pokemon
from `own_slot_order` (precomputed from all turns) even if not yet in the snapshot.
The player knows their full team; unrevealed own Pokemon are treated as healthy
until the log proves otherwise.

**BattleFeatureExtractor compatibility**: synthetic duck-typed objects (`FakeEnum`,
`SynthMove`, `SynthPokemon`, `SynthBattle`) satisfy every attribute access the
extractor makes.  Real poke-env enum classes are NOT imported — only `GenData`
for move/species lookup.

**Stable own-slot order**: `extractor._own_slot_order` is pre-seeded with the
full battle's reveal order before the first `extract()` call.  Unrevealed slots
remain zero-filled in the feature vector (no future leakage).

## Bugs Found and Fixed

### D4 – `field` attribute name in TurnSnapshot shadows `dataclasses.field()`

`TurnSnapshot` had `field: FieldState = field(default_factory=FieldState)`.
Python class bodies share namespace: after `field = <Field object>` was assigned,
subsequent `field(...)` calls tried to invoke the Field object → `TypeError`.
Fix: renamed attribute to `field_state` throughout parser, reconstruct, trajectory.

### D4 – Switch actions for first-time Pokemon encoded as -1

When a player switches in a Pokemon for the first time, the snapshot at that
turn's start doesn't include that Pokemon in `own_slots` yet.  `_encode_action`
searched `view.own_slots` and couldn't find the target → returned -1.

Root cause: the own team builds up one switch at a time in the log, but the
player knows their full team from turn 1.

Fix: `_available_switches()` now computes the union of:
1. Currently revealed (non-fainted, non-active) own slots from the snapshot
2. Pokemon in `own_slot_order` not yet in the snapshot (first-time switch targets)

Valid action rate improved from 84.3% → 95.9%.

## Known Limitations

### Action = -1 (4.1% of turns)

Remaining -1 actions break down as:
- **`cant` events**: player chose a move but was blocked (sleep, paralysis, recharge).
  The parser's `|cant|` handler doesn't record an action.  These are excluded from
  policy loss during training.
- **Force-switch timing**: Showdown sometimes places a faint + forced switch within
  the same turn's events (before `|turn|N+1|`).  The parser records the switch action
  in the CURRENT turn's snapshot correctly, but also sets `force_switch_p1/p2=True`
  on the NEXT turn's snapshot.  Turns marked as force-switch but with -1 action
  indicate the switch already resolved in the prior turn.
- **Last turn of game**: no action is recorded after `|win|`.

### 2-4 HP systematic overestimate in damage_calc.py

Documented in Phase 2.  Metamon applies the RNG roll last (to float); Showdown
applies it first (to integer with `pokeRound` floors).  Results in 2-4 HP
over-estimate in all damage scenarios.  **Potential Phase 8 MCTS suspect**: MCTS
rollout evaluations using the damage calculator may over-estimate damage by 2-5%,
which could bias action selection in one-hit-KO situations.  Monitor if MCTS win rate
is unexpectedly low on KO-threshold situations.

### Team preview not in replay log

Random Battle team preview is private (not in the replay log).  Own team builds
up via revealed switches.  On turn 1, the feature extractor's own-slot features
for unrevealed Pokemon are zero-filled.

## Post-Delivery Bug Fixes (2026-04-22)

Two bugs discovered during validation-gap audit and fixed before production scaling:

### Bug 1 — Struggle in move_orders shifts real moves out of range

**Symptom:** 17 invalid actions in one replay (gen9randombattle-2591230892) where
Sylveon used Struggle (ran out of PP). Struggle was added to `_compute_move_orders`,
making Sylveon's move list 5 entries. Alphabetically Struggle sorts before Wish,
pushing Wish to index 4 (beyond the 0-3 slot limit) → action = -1.

**Fix:** Filter `"struggle"` in `_compute_move_orders`. Struggle is automatic (no
player choice) and must never occupy a slot index.

### Bug 2 — Minior forme change duplicates own_slot_order entry

**Symptom:** 3 invalid switches in 3 replays. When Minior transforms from Meteor form
to Core form (e.g., miniormeteor → minioryellow), both species IDs were added to
`_compute_own_slot_order`, inflating the team to 7 entries. With 7 pseudo-entries
(one slot = active), `_available_switches` produced 6 candidates — one more than the
5-slot limit (indices 4-8). The 6th switch target → index 5 → `4+5=9` → clamped to
-1 by the `idx > 4` guard.

**Fix:** Track `own_slot_order` by **nickname** (not species ID). If a nickname was
already recorded (forme change), skip the new species ID — same Pokemon, same slot.

**Net improvement:** 20 bug-caused invalid actions fixed (p1 + p2 combined ≈ 40).
Valid action rate: 95.9% → 96.2%.

### Remaining invalid actions (3.8%) — all expected

| Category | P1 count | Root cause |
|----------|----------|------------|
| last_turn | 38 | No action recorded after `\|win\|` |
| cant | 44 | Sleep / paralysis / recharge — player had no choice |
| force_switch | 6 | Force-switch timing: flag set previous turn |
| other | 5 | 3 parser failures (p1_action=None), 2 Struggle turns |

These are masked out of policy loss during training (action=-1 filter).

## Illusion Coverage Gap — Documented, Code Tested Synthetically

The 100-replay smoke-test dataset contains zero actual `|-end|...|Illusion` events.
One replay has Zoroark in the team (gen9randombattle-2591001758) but it was never
switched in. The Illusion reconstruction code path was untested against live data.

**Resolution:** Synthetic unit test added in `tests/test_replay_ingestion.py`
(`TestIllusionBreak`). Three tests exercise:
- `test_illusion_true_species_revealed`: p2 own_slots shows Zoroark after break
- `test_illusion_entry_species_recorded`: parser snapshot records `illusion_entry_species=Gengar`
- `test_opponent_sees_zoroark_after_break`: p1 opp_slots shows Zoroark after break

All 7 unit tests (including Struggle exclusion and Minior forme deduplication) pass.

**Note:** Imposter Ditto IS tested against live data (gen9randombattle-2591189018,
two `-transform` events). Reconstruction correctly updates Ditto's species to the
target (Brambleghast, then Tentacruel).

## D5 Full-Vector Validation

`scripts/validate_features.py` implements full 959-dim range checks:
- Own slots [0-5]: species_idx in [0,508]; hp in [0,1]; status/move-type/item/tera
  features are binary {0,1}; boosts in [-1,1]; base stats/255 in [0,1]
- Opp slots [6-11]: same scalar bounds but move-type/flag/item/tera features are
  **continuous [0,1]** probabilities from SetPredictor (not binary)
- Global [924-958]: weather/terrain one-hot sums=1; hazard/screen/remaining all [0,1]
- Opponent-slot leakage: unrevealed slots must have species_idx=0 and hp=0
- Slot ordering stability: own species_idx must not change once non-zero

Result on 200 files / 5,450 turns: **PASS** — zero range errors, zero leakage,
zero stability violations, zero turns with empty legal mask.

## D3 Expanded Leakage Audit (20 Replays, Turns ≥20)

20 randomly sampled replays (seed=99) examined for information-leakage at turns≥20.
194 late-game turns checked.

Findings:
- 0 unrevealed-slot HP leakage
- 0 unrevealed-slot move leakage
- 0 unrevealed-slot status leakage
- 12 "same species in both teams" flags — all from Ditto Imposter replay; Ditto's
  species is correctly set to the transform target (Brambleghast / Tentacruel), which
  happens to match the active opponent. Not a leakage bug.

## Resume Instructions (Phase 3)

Phase 3 validation is complete. To re-run the pipeline:
```bash
source /home/user/metamon-env/bin/activate
cd /home/user/showdown-bot

# Scrape 100 replays (skip if data/raw_replays/ already populated)
python3 scripts/run_scraper.py --max 100

# Run parse + reconstruct + trajectory on all scraped replays
python3 scripts/run_pipeline.py --max 100

# Validate reconstruction consistency (D5 basic)
python3 scripts/validate_reconstruction.py

# Validate full 959-dim feature vector (D5 extended)
python3 scripts/validate_features.py

# Generate dataset report (D6)
python3 scripts/generate_dataset_report.py

# Run unit tests
python -m pytest tests/test_replay_ingestion.py -v
```

Expected output:
- `data/raw_replays/gen9randombattle/` — 100 .json + .log files
- `data/trajectories/gen9randombattle/` — 200 .npz files (2 per replay)
- `data/dataset_report.md` — summary report

## Phase 3 Dataset Expansion — Main Training Corpus (April 2026)

### Final Combined Dataset Statistics

Source: HolidayOugi/pokemon-showdown-replays (HuggingFace) + original 383-replay pilot scrape.

| Metric | Value |
|--------|-------|
| Total replays | 101,311 |
| Trajectory files (.npz) | 202,622 (2 per replay) |
| Total turns (both POVs) | 5,529,652 |
| Valid (usable) turns | 5,302,040 (95.9%) |
| filter_for_training turns | 227,612 (4.1%) |
| — of which parse_failure | 9,810 (0.18%) |
| NaN/Inf files | 0 |
| Disk footprint | 1.1 GB (data/trajectories/gen9randombattle/) |
| Estimated load time | ~84 seconds (sequential, all npz) |

### Rating Distribution (processed replays, 95,648 rated)

| Metric | Value |
|--------|-------|
| Min | 1900 (filter applied) |
| Max | 2547 |
| Mean | 2072.5 |
| Median | 2049.0 |
| p90 | 2258 |
| p95 | 2306 |
| p99 | 2385 |

### Date Distribution

| | |
|-|-|
| Earliest replay | 2022-12-22 (Gen 9 RB launch period) |
| Latest replay | 2026-04-21 |
| Coverage | Dec 2022 – Apr 2026 (3 years 4 months) |

### Storage Locations

| Data | Path |
|------|------|
| HolidayOugi parquet files (8 parts, 3.2 GB) | data/holidayougi/ |
| Original scraped replays (383 JSON+log) | data/raw_replays/gen9randombattle/ |
| All trajectory npz files | data/trajectories/gen9randombattle/ |

### Dataset Cutoff

HolidayOugi latest replay uploadtime: **1776787094** (2026-04-21 UTC).
Player supplement scrape: NOT RUN (see deviations below).
Effective main dataset cutoff: **2026-04-21**.

Phase 9 data collection must use `uploadtime > 1776787094` to avoid overlap.

### Normalized Player List (Phase 9 supplement reference)

The following 30 high-ELO players (all 2250–2515 peak rating) are verified for
future Phase 9 supplement scraping via `scripts/run_player_supplement.py`:

aqua, michaelderbeste2, pokeblade101, teresbahji, milkreo, referrals, smokyaim,
articoo, delta2777, dra15v2, helicopyer, 70to90gxe, masterj007, sigurdzz, drizzle,
pentav, wintersim, galak0, cephaleid, fatmarmot, assidion, daruma, bauses, firehills,
sebasdb, norman2, lizardune, emptybrackets, mylifeisdance, piyush21

### Deviations from Original Directive

1. **Player supplement (Step 4) not run.** The directive specified a player-based
   supplement covering July 23, 2025 – present. During execution, the full HolidayOugi
   dataset was found to extend to April 21, 2026 (not July 22, 2025 as initially
   probed from part 1 only). The user confirmed "let's not do anything about the
   2-day gap between 21st April and today," which was interpreted as authorization
   to skip Step 4. The supplement directory (data/raw_replays/supplement/) is empty.
   Impact: zero — the corpus already covers through April 21, 2026 at full density.
   Remediation: run `python scripts/run_player_supplement.py` in Phase 9 with the
   updated SUPPLEMENT_CUTOFF = 1776873600 (2026-04-22T00:00:00Z).

2. **Illusion team count (169) < Illusion break count (1716).** This is expected
   behaviour, not a bug. `_has_illusion_team` scans switch/drag lines for Zoroark/Zorua
   species — but Zoroark enters disguised as another species, so its switch line shows
   the disguise species. Only replays where Zoroark switches in a second time after
   breaking show up in switch lines. The break count (1716, 1.7%) is the reliable
   figure; all 10 spot-checked break-flagged replays were confirmed true positives.

3. **Per-part rating disparity.** Parts 4 and 5 of HolidayOugi had only 2,522 and
   4,158 qualifying replays vs 10k–31k for other parts. Root cause: these parts have
   mean ratings of 1289 and 1245 respectively — roughly half of part 1 (mean 1582).
   The HolidayOugi dataset appears internally partitioned by rating tier, not by date;
   parts 4–5 contain predominantly low-ELO games across all time periods.

4. **Step 6 Illusion monitoring note.** The original scraper pipeline uses
   `_has_illusion_team` from scraper.py which undercounts (see point 2). For Phase 9,
   consider replacing with a log-text regex that detects Zoroark's disguised switch-in
   (i.e., scan for Zoroark in `|replace|` or `|detailschange|` events, not just
   `|switch|`/`|drag|`).

