#!/usr/bin/env python3
"""
Phase 3 pipeline runner: parse → reconstruct → trajectory for all scraped replays.

Processes all .json files in the raw replay directory and writes .npz trajectory
files to the output directory.  Prints per-batch progress and a final summary.

Usage:
  cd /home/user/showdown-bot
  python scripts/run_pipeline.py [--max 1000] [--workers 1]

Acceptance criteria (smoke test at 100 replays):
  - Parse failure rate  < 1%   (winner detected in ≥ 99 of 100)
  - Reconstruction failures < 2%
  - Trajectory failure rate  < 2%
  - No NaN / Inf in any state vector
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/user/metamon")

import os
os.environ.setdefault("METAMON_ALLOW_ANY_POKE_ENV", "True")
os.environ.setdefault("METAMON_CACHE_DIR", "/home/user/metamon-cache")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_pipeline")


def process_file(json_path: Path, trajectory_dir: Path, builder) -> dict:
    """Process one replay file through parse → reconstruct → trajectory."""
    from replay_ingestion.parser import parse_replay_file
    from replay_ingestion.reconstruct import reconstruct

    result = {"path": str(json_path), "ok": False, "turns": 0, "winner": None, "error": None}

    try:
        battle = parse_replay_file(str(json_path))
        if battle is None:
            result["error"] = "parse_returned_none"
            return result
        if not battle.turns:
            result["error"] = "no_turns"
            return result

        rec = reconstruct(battle)
        if rec is None:
            result["error"] = "reconstruct_returned_none"
            return result

        stats = builder.build_and_save(rec, trajectory_dir)
        result["ok"] = stats["npz_files_saved"] == 2
        result["turns"] = stats["turns"]
        result["winner"] = stats["winner"]
        if stats["errors"]:
            result["error"] = "; ".join(stats["errors"])
    except Exception as exc:
        result["error"] = str(exc)
        logger.exception("Pipeline error on %s", json_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 3 pipeline runner")
    parser.add_argument("--max", type=int, default=None, help="Max replays to process")
    parser.add_argument(
        "--replay-dir",
        default="data/raw_replays/gen9randombattle",
    )
    parser.add_argument(
        "--output-dir",
        default="data/trajectories/gen9randombattle",
    )
    parser.add_argument("--check-nan", action="store_true", default=True,
                        help="Check NaN/Inf in state vectors (default: on)")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    replay_dir = base / args.replay_dir
    traj_dir = base / args.output_dir

    json_files = sorted(replay_dir.glob("*.json"))
    if args.max is not None:
        json_files = json_files[: args.max]

    print(f"Phase 3 Pipeline Runner")
    print(f"  Replay directory : {replay_dir}")
    print(f"  Output directory : {traj_dir}")
    print(f"  Replays to process: {len(json_files)}")
    print()

    if not json_files:
        print("No replay files found. Run scripts/run_scraper.py first.")
        return 1

    from replay_ingestion.trajectory import TrajectoryBuilder
    builder = TrajectoryBuilder()

    t_start = time.time()
    results = []
    nan_count = 0

    for i, json_path in enumerate(json_files):
        r = process_file(json_path, traj_dir, builder)
        results.append(r)

        # NaN check on saved files
        if r["ok"] and args.check_nan:
            rid = json_path.stem
            for p in (1, 2):
                npz_path = traj_dir / f"{rid}_p{p}.npz"
                if npz_path.exists():
                    try:
                        data = np.load(npz_path)
                        if np.any(np.isnan(data["states"])) or np.any(np.isinf(data["states"])):
                            nan_count += 1
                            logger.warning("NaN/Inf in %s", npz_path)
                    except Exception:
                        pass

        if (i + 1) % 10 == 0 or (i + 1) == len(json_files):
            ok = sum(1 for r in results if r["ok"])
            elapsed = time.time() - t_start
            print(f"  [{i+1:4d}/{len(json_files)}] OK={ok}  t={elapsed:.1f}s")

    elapsed = time.time() - t_start

    # Summary
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    winner_detected = sum(1 for r in ok if r["winner"] in (1, 2))
    total_turns = sum(r["turns"] for r in ok)

    print()
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total replays   : {len(results)}")
    print(f"  Succeeded       : {len(ok)}  ({100*len(ok)/len(results):.1f}%)")
    print(f"  Failed          : {len(failed)}  ({100*len(failed)/len(results):.1f}%)")
    print(f"  Winner detected : {winner_detected} / {len(ok)}")
    print(f"  Total turns     : {total_turns}")
    print(f"  NaN/Inf files   : {nan_count}")
    print(f"  Wall time       : {elapsed:.1f}s")
    if ok:
        print(f"  Avg turns/battle: {total_turns/len(ok):.1f}")
        print(f"  Throughput      : {len(results)/elapsed:.2f} battles/s")
    print()

    if failed:
        print("Failure breakdown:")
        err_counts: dict = {}
        for r in failed:
            key = r.get("error") or "unknown"
            err_counts[key] = err_counts.get(key, 0) + 1
        for err, cnt in sorted(err_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cnt:4d}x  {err[:80]}")
        print()

    # Acceptance verdict
    fail_rate = len(failed) / len(results) if results else 1.0
    passed = fail_rate < 0.02 and nan_count == 0
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        if fail_rate >= 0.02:
            print(f"  ✗ Failure rate {fail_rate:.1%} ≥ 2%")
        if nan_count > 0:
            print(f"  ✗ {nan_count} files with NaN/Inf")
    print()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
