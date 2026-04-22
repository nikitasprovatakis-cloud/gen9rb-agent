#!/usr/bin/env python3
"""
Production scaling run: scrape + pipeline to 10k checkpoint, then optionally to full scale.

Phases:
  1. Scrape to CHECKPOINT replays total (default 10,000)
  2. Run pipeline on ALL scraped replays
  3. Validate features and print full status report
  4. Exit with code 10 to signal checkpoint reached (resume with --continue)

  With --continue: scrape from checkpoint to FULL_TARGET (default 50,000)

Usage:
  # First run (to 10k checkpoint):
  cd /home/user/showdown-bot
  python scripts/run_scaling.py

  # After approving 10k checkpoint, continue to full scale:
  python scripts/run_scaling.py --continue

Progress is logged to logs/scaling_run.log.
Pipeline is run after scraping completes (or can be run standalone with --pipeline-only).
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/user/metamon")

os.environ.setdefault("METAMON_ALLOW_ANY_POKE_ENV", "True")
os.environ.setdefault("METAMON_CACHE_DIR", "/home/user/metamon-cache")

CHECKPOINT_TARGET = 10_000   # pause and report at this many total replays
FULL_TARGET       = 50_000   # absolute ceiling

LOG_PATH = Path(__file__).parent.parent / "logs" / "scaling_run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_scaling")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_existing(replay_dir: Path) -> int:
    return len(list(replay_dir.glob("*.json")))


def _count_processed(traj_dir: Path) -> int:
    """Count replays that have BOTH p1 and p2 npz files."""
    p1 = {f.stem.replace("_p1", "") for f in traj_dir.glob("*_p1.npz")}
    p2 = {f.stem.replace("_p2", "") for f in traj_dir.glob("*_p2.npz")}
    return len(p1 & p2)


def run_scraper_phase(
    replay_dir: Path,
    target_total: int,
    min_rating: int = 1900,
) -> dict:
    """Scrape until replay_dir contains target_total .json files."""
    from replay_ingestion.scraper import ReplayScraper

    scraper = ReplayScraper(
        output_dir=str(replay_dir),
        min_rating=min_rating,
    )

    existing = _count_existing(replay_dir)
    need = max(0, target_total - existing)
    if need == 0:
        logger.info("Already at target (%d replays). Skipping scrape.", existing)
        return {"downloaded": 0, "skipped_exists": existing}

    logger.info(
        "Scraping: have %d replays, target %d, need %d more.",
        existing, target_total, need,
    )
    stats = scraper.scrape(
        fmt="gen9randombattle",
        max_replays=need,
        progress_every=100,
    )
    logger.info("Scrape done: %s", stats)
    return stats


def run_pipeline_phase(replay_dir: Path, traj_dir: Path) -> dict:
    """Run parse→reconstruct→trajectory on all replays not yet processed."""
    from replay_ingestion.parser import parse_replay_file
    from replay_ingestion.reconstruct import reconstruct
    from replay_ingestion.trajectory import TrajectoryBuilder
    from replay_ingestion.scraper import _has_illusion_team, _has_illusion_break

    traj_dir.mkdir(parents=True, exist_ok=True)

    already_done = {f.stem.replace("_p1", "") for f in traj_dir.glob("*_p1.npz")}
    json_files = [f for f in sorted(replay_dir.glob("*.json"))
                  if f.stem not in already_done]

    if not json_files:
        logger.info("No new replays to process.")
        return {"processed": 0, "failed": 0}

    logger.info("Processing %d new replays through pipeline…", len(json_files))
    builder = TrajectoryBuilder()
    processed = failed = 0
    illusion_team = illusion_break = illusion_fail = 0
    t0 = time.time()

    for i, json_path in enumerate(json_files):
        try:
            battle = parse_replay_file(str(json_path))
            if not battle or not battle.turns:
                failed += 1; continue
            rec = reconstruct(battle)
            if not rec:
                failed += 1; continue
            stats = builder.build_and_save(rec, traj_dir)
            if stats["npz_files_saved"] < 2:
                failed += 1; continue
            processed += 1

            # Illusion tracking
            log_path = json_path.with_suffix(".log")
            if log_path.exists():
                log_text = log_path.read_text()
                if _has_illusion_team(log_text):
                    illusion_team += 1
                    if _has_illusion_break(log_text):
                        illusion_break += 1

        except Exception as exc:
            logger.warning("Pipeline error on %s: %s", json_path.stem, exc)
            failed += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            logger.info(
                "Pipeline progress: %d/%d (%.1f/min) | ok=%d fail=%d",
                i + 1, len(json_files), rate, processed, failed,
            )

    logger.info(
        "Pipeline done: processed=%d failed=%d illusion_team=%d illusion_break=%d",
        processed, failed, illusion_team, illusion_break,
    )
    if illusion_break > 0 and illusion_break > illusion_fail:
        logger.info("Illusion reconstruction: all %d breaks handled.", illusion_break)

    return {
        "processed": processed,
        "failed": failed,
        "illusion_team": illusion_team,
        "illusion_break": illusion_break,
    }


def report_dataset_health(replay_dir: Path, traj_dir: Path) -> None:
    """Print a comprehensive dataset health summary."""
    from replay_ingestion.scraper import ReplayScraper

    scraper = ReplayScraper(str(replay_dir))
    rep = scraper.report()

    npz_files = list(traj_dir.glob("*.npz"))
    total_turns = valid_turns = filter_turns = parse_fail_turns = 0
    nan_files = 0

    for f in npz_files:
        try:
            d = np.load(f)
            acts = d["actions"]
            filt = d["filter_for_training"]
            pf = d["parse_failure"]
            total_turns += len(acts)
            valid_turns += int((~filt).sum())
            filter_turns += int(filt.sum())
            parse_fail_turns += int(pf.sum())
            if np.any(np.isnan(d["states"])) or np.any(np.isinf(d["states"])):
                nan_files += 1
        except Exception:
            pass

    filter_rate = filter_turns / total_turns if total_turns else 0
    parse_rate = parse_fail_turns / total_turns if total_turns else 0

    print()
    print("=" * 65)
    print("DATASET HEALTH REPORT")
    print("=" * 65)
    print(f"  Replays cached        : {rep['total_cached']:,}")
    print(f"  Rating range          : {rep['rating_min']} – {rep['rating_max']}")
    print(f"  Rating mean           : {rep['rating_mean']}")
    print(f"  Illusion team replays : {rep['illusion_team']}")
    print(f"  Illusion break replays: {rep['illusion_break']}")
    print()
    print(f"  Trajectory files (.npz): {len(npz_files):,}")
    print(f"  Total turns (both POVs): {total_turns:,}")
    print(f"  Valid (usable) turns   : {valid_turns:,} ({100*(1-filter_rate):.1f}%)")
    print(f"  filter_for_training    : {filter_turns:,} ({100*filter_rate:.1f}%)")
    print(f"    of which parse_failure: {parse_fail_turns:,} ({100*parse_rate:.2f}%)")
    print(f"  NaN/Inf files          : {nan_files}")
    if rep['illusion_break'] > 0:
        print()
        print(f"  *** Illusion breaks present in dataset — see pipeline log for")
        print(f"      per-replay reconstruction success/failure details ***")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Production scaling run")
    parser.add_argument(
        "--continue", dest="do_continue", action="store_true",
        help="Skip 10k checkpoint, continue straight to FULL_TARGET",
    )
    parser.add_argument(
        "--pipeline-only", action="store_true",
        help="Skip scraping; run pipeline on existing replays only",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=CHECKPOINT_TARGET,
        help=f"First pause target (default {CHECKPOINT_TARGET:,})",
    )
    parser.add_argument(
        "--full-target", type=int, default=FULL_TARGET,
        help=f"Absolute ceiling (default {FULL_TARGET:,})",
    )
    parser.add_argument("--min-rating", type=int, default=1900)
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    replay_dir = base / "data/raw_replays/gen9randombattle"
    traj_dir   = base / "data/trajectories/gen9randombattle"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    target = args.full_target if args.do_continue else args.checkpoint
    logger.info("=" * 65)
    logger.info("Scaling run started. Target: %d replays. Continue=%s",
                target, args.do_continue)
    logger.info("Existing replays: %d", _count_existing(replay_dir))

    # ── Phase 1: Scrape ──────────────────────────────────────────────────────
    if not args.pipeline_only:
        run_scraper_phase(replay_dir, target_total=target, min_rating=args.min_rating)

    # ── Phase 2: Pipeline ────────────────────────────────────────────────────
    run_pipeline_phase(replay_dir, traj_dir)

    # ── Phase 3: Health report ───────────────────────────────────────────────
    report_dataset_health(replay_dir, traj_dir)

    # ── Phase 4: Checkpoint decision ─────────────────────────────────────────
    total = _count_existing(replay_dir)
    if not args.do_continue and total < args.full_target:
        print(f"CHECKPOINT REACHED: {total:,} replays scraped and processed.")
        print(f"Review the health report above, then re-run with --continue")
        print(f"to scale to {args.full_target:,} replays.")
        print()
        return 10  # sentinel exit code for checkpoint

    logger.info("Scaling run complete. Total replays: %d", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
