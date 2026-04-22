#!/usr/bin/env python3
"""
Phase 1 Deliverable 3: Run Metamon Gen 9 checkpoint self-play in Gen 9 Random Battle.
Two instances of SmallRLGen9Beta play 1 battle each against each other on the local server.
Acceptance criteria:
- A Gen 9 Random Battle completes between two instances
- Valid move selections (no forfeit-by-timeout, no illegal action errors)
- Completion logged with winner, turn count, wall-clock duration
"""

import os
import sys
import time
import json
import datetime
import multiprocessing

BASE_DIR = "/home/user"
METAMON_DIR = f"{BASE_DIR}/metamon"
LOG_DIR = f"{BASE_DIR}/showdown-bot/logs"

os.makedirs(LOG_DIR, exist_ok=True)


def _env_setup():
    os.environ["METAMON_ALLOW_ANY_POKE_ENV"] = "True"
    os.environ["METAMON_CACHE_DIR"] = f"{BASE_DIR}/metamon-cache"
    if METAMON_DIR not in sys.path:
        sys.path.insert(0, METAMON_DIR)


def _patch_format_aliases():
    import metamon.config
    metamon.config.FORMAT_ALIASES["gen9randombattle"] = "gen9ou"


def _build_cpu_model():
    """SmallRLGen9Beta with FlashAttention override removed so VanillaAttention is used."""
    import amago.nets.transformer
    from metamon.rl.pretrained import SmallRLGen9Beta
    model = SmallRLGen9Beta()
    # gin_overrides = None means base_config will auto-select VanillaAttention (CPU-safe)
    model.gin_overrides = None
    return model


def run_player(username, n_battles, result_queue):
    """Target for each worker process."""
    import warnings
    warnings.filterwarnings("ignore")

    _env_setup()

    import amago
    import metamon
    _patch_format_aliases()

    from metamon.rl.evaluate import pretrained_vs_local_ladder

    model = _build_cpu_model()
    start = time.time()
    try:
        results = pretrained_vs_local_ladder(
            pretrained_model=model,
            username=username,
            battle_format="gen9randombattle",
            team_set=None,   # server provides team for random battle
            total_battles=n_battles,
            battle_backend="pokeagent",
            log_to_wandb=False,
            save_trajectories_to=None,
            save_team_results_to=None,
        )
        elapsed = time.time() - start
        result_queue.put({
            "username": username,
            "status": "ok",
            "results": results,
            "elapsed_s": round(elapsed, 1),
        })
    except Exception as exc:
        elapsed = time.time() - start
        result_queue.put({
            "username": username,
            "status": "error",
            "error": str(exc),
            "elapsed_s": round(elapsed, 1),
        })


def main():
    _env_setup()
    _patch_format_aliases()

    print("=" * 60)
    print("Phase 1 Deliverable 3 – Gen 9 Random Battle self-play")
    print("Checkpoint: SmallRLGen9Beta (VanillaAttention)")
    print("=" * 60)

    # Pre-download the checkpoint so both workers don't race to download it
    print("\nDownloading SmallRLGen9Beta checkpoint (first run only)...")
    model = _build_cpu_model()
    ckpt_path = model.get_path_to_checkpoint(model.default_checkpoint)
    print(f"Checkpoint cached at: {ckpt_path}\n")

    n_battles = 1
    result_queue = multiprocessing.Queue()

    p1 = multiprocessing.Process(
        target=run_player,
        args=("MetaPlayer1", n_battles, result_queue),
        daemon=True,
    )
    p2 = multiprocessing.Process(
        target=run_player,
        args=("MetaPlayer2", n_battles, result_queue),
        daemon=True,
    )

    wall_start = time.time()
    print("Starting both players simultaneously...")
    p1.start()
    # small stagger so Player 1 is queued before Player 2 joins
    time.sleep(3)
    p2.start()

    timeout_s = 1800  # 30-minute timeout (GPU inference is fast, but model init takes time)
    results = []
    for _ in range(2):
        try:
            r = result_queue.get(timeout=timeout_s)
            results.append(r)
        except Exception:
            print("ERROR: Timed out waiting for a player result")
            break

    p1.join(timeout=10)
    p2.join(timeout=10)

    wall_elapsed = time.time() - wall_start

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    errors = [r for r in results if r.get("status") == "error"]
    oks = [r for r in results if r.get("status") == "ok"]

    if errors:
        for e in errors:
            print(f"  {e['username']} ERROR: {e['error']}")

    for r in oks:
        print(f"  {r['username']}: {r.get('results', {})}")
        print(f"  Elapsed: {r['elapsed_s']}s")

    print(f"\nTotal wall-clock: {wall_elapsed:.1f}s")

    # Write log entry
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "deliverable": "d3_selfplay",
        "format": "gen9randombattle",
        "checkpoint": "SmallRLGen9Beta",
        "attention": "VanillaAttention",
        "n_battles_per_player": n_battles,
        "wall_clock_s": round(wall_elapsed, 1),
        "player_results": results,
    }
    log_path = os.path.join(LOG_DIR, "d3_selfplay.json")
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)
    print(f"\nLog written to {log_path}")

    any_error = any(r.get("status") == "error" for r in results)
    if any_error or len(results) < 2:
        print("\nDELIVERABLE 3: FAIL (see errors above)")
        sys.exit(1)
    else:
        print("\nDELIVERABLE 3: PASS")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Required for CUDA; fork + CUDA = deadlocks
    main()
