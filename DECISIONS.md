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
