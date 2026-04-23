#!/usr/bin/env python3
"""
Step 5 (HolidayOugi portion): Parse and build trajectories from the
HolidayOugi parquet dataset.

Processes all parquet parts, filtering for:
  - rating >= 1900  (nulls excluded)
  - OR smogtours-* replay IDs (tournament games, any rating)

Writes npz trajectory files to data/trajectories/gen9randombattle/.
Resume-safe: skips replays whose npz files already exist.

Usage:
  python scripts/process_holidayougi.py [--max N] [--part 1]
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

LOG_PATH = Path(__file__).parent.parent / "logs" / "process_holidayougi.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("process_holidayougi")

MIN_RATING = 1900


def process_row(row, traj_dir: Path, builder) -> dict:
    """Process one parquet row through parse → reconstruct → trajectory."""
    from replay_ingestion.parser import Gen9Parser
    from replay_ingestion.reconstruct import reconstruct

    rid        = row["id"]
    log_text   = row["log"]
    uploadtime = int(row["uploadtime"])
    rating     = row.get("rating")

    result = {"id": rid, "ok": False, "turns": 0, "winner": None,
              "error": None, "filter_count": 0, "parse_failure_count": 0,
              "illusion_team": False, "illusion_break": False}

    # Check if already processed
    p1_file = traj_dir / f"{rid}_p1.npz"
    p2_file = traj_dir / f"{rid}_p2.npz"
    if p1_file.exists() and p2_file.exists():
        result["ok"] = True
        result["skipped_exists"] = True
        return result

    try:
        from replay_ingestion.scraper import _has_illusion_team, _has_illusion_break
        result["illusion_team"] = _has_illusion_team(log_text)
        result["illusion_break"] = _has_illusion_break(log_text)

        parser = Gen9Parser()
        battle = parser.parse(log_text, replay_id=rid, upload_time=uploadtime)

        if battle is None:
            result["error"] = "parse_returned_none"; return result
        if not battle.turns:
            result["error"] = "no_turns"; return result

        rec = reconstruct(battle)
        if rec is None:
            result["error"] = "reconstruct_returned_none"; return result

        stats = builder.build_and_save(rec, traj_dir)
        if stats["npz_files_saved"] < 2:
            result["error"] = f"only {stats['npz_files_saved']} npz saved"
            return result

        result["ok"]     = True
        result["turns"]  = stats["turns"]
        result["winner"] = stats["winner"]
        if stats.get("errors"):
            result["error"] = "; ".join(stats["errors"])

        for pov in (1, 2):
            npz_path = traj_dir / f"{rid}_p{pov}.npz"
            if npz_path.exists():
                d = np.load(npz_path)
                result["filter_count"]        += int(d["filter_for_training"].sum())
                result["parse_failure_count"] += int(d["parse_failure"].sum())

    except Exception as exc:
        result["error"] = str(exc)[:200]
        logger.warning("Pipeline error on %s: %s", rid, exc)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None,
                        help="Max rows to process (across all parts, for testing)")
    parser.add_argument("--part", type=int, default=None,
                        help="Process only this part number (1-8)")
    parser.add_argument("--min-rating", type=int, default=MIN_RATING)
    args = parser.parse_args()

    base           = Path(__file__).parent.parent
    parquet_dir    = base / "data" / "holidayougi"
    traj_dir       = base / "data" / "trajectories" / "gen9randombattle"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Select parquet parts
    if args.part:
        parts = [parquet_dir / f"part{args.part}.parquet"]
    else:
        parts = sorted(parquet_dir.glob("part*.parquet"))

    if not parts:
        print("No parquet files found in", parquet_dir)
        return 1

    logger.info("=" * 65)
    logger.info("HolidayOugi processing started. Parts: %d  min_rating=%d",
                len(parts), args.min_rating)

    import pandas as pd
    from replay_ingestion.trajectory import TrajectoryBuilder

    builder  = TrajectoryBuilder()
    t_global = time.time()

    total_rows_read   = 0
    total_qualifying  = 0
    total_processed   = 0
    total_failed      = 0
    total_skipped     = 0
    total_turns       = 0
    total_filter      = 0
    total_parse_fail  = 0
    total_illusion_t  = 0
    total_illusion_b  = 0
    nan_count         = 0

    for part_path in parts:
        logger.info("Loading %s ...", part_path.name)
        df = pd.read_parquet(part_path)
        total_rows_read += len(df)

        # Filter: rating >= min_rating OR smogtours tournament replay
        mask_rated  = df["rating"].notna() & (df["rating"] >= args.min_rating)
        mask_tourn  = df["id"].str.startswith("smogtours-")
        df_filt     = df[mask_rated | mask_tourn].copy()
        qualifying  = len(df_filt)
        total_qualifying += qualifying

        logger.info("  Part %s: %d rows → %d qualifying (rated≥%d or smogtours)",
                    part_path.name, len(df), qualifying, args.min_rating)

        if args.max is not None:
            remaining = args.max - total_processed - total_skipped
            if remaining <= 0:
                break
            df_filt = df_filt.head(remaining)

        part_processed = part_failed = part_skipped = 0
        t_part = time.time()

        for i, (_, row) in enumerate(df_filt.iterrows()):
            r = process_row(row, traj_dir, builder)

            if r.get("skipped_exists"):
                part_skipped  += 1
                total_skipped += 1
                continue

            if r["ok"]:
                part_processed  += 1
                total_processed += 1
                total_turns     += r["turns"]
                total_filter    += r["filter_count"]
                total_parse_fail += r["parse_failure_count"]
                if r["illusion_team"]:
                    total_illusion_t += 1
                if r["illusion_break"]:
                    total_illusion_b += 1

                # Spot NaN check (every 1000th replay to keep speed)
                if part_processed % 1000 == 0:
                    rid = r["id"]
                    for pov in (1, 2):
                        npz_p = traj_dir / f"{rid}_p{pov}.npz"
                        if npz_p.exists():
                            d = np.load(npz_p)
                            if np.any(np.isnan(d["states"])) or np.any(np.isinf(d["states"])):
                                nan_count += 1
                                logger.warning("NaN/Inf in %s", npz_p)
            else:
                part_failed  += 1
                total_failed += 1

            if (i + 1) % 500 == 0:
                elapsed    = time.time() - t_part
                rate       = (i + 1) / elapsed * 60
                logger.info(
                    "  [%s] %d/%d  ok=%d fail=%d skip=%d  %.1f/min",
                    part_path.name, i + 1, len(df_filt),
                    part_processed, part_failed, part_skipped, rate,
                )

        part_elapsed = time.time() - t_part
        logger.info(
            "  Part %s done: ok=%d fail=%d skip=%d  %.1fs",
            part_path.name, part_processed, part_failed, part_skipped, part_elapsed,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_global
    total_turns_both = total_turns * 2
    filter_rate = total_filter / total_turns_both if total_turns_both else 0
    parse_rate  = total_parse_fail / total_turns_both if total_turns_both else 0

    print()
    print("=" * 65)
    print("HOLIDAYOUGI PROCESSING SUMMARY")
    print("=" * 65)
    print(f"  Parquet rows read     : {total_rows_read:,}")
    print(f"  Qualifying rows       : {total_qualifying:,}")
    print(f"  Successfully processed: {total_processed:,}")
    print(f"  Already existed (skip): {total_skipped:,}")
    print(f"  Failed                : {total_failed:,}")
    fail_rate = total_failed / (total_processed + total_failed) if (total_processed + total_failed) else 0
    print(f"  Failure rate          : {100*fail_rate:.2f}%")
    print()
    print(f"  Total turns (both POV): {total_turns_both:,}")
    print(f"  Valid (usable) turns  : {total_turns_both - total_filter:,} ({100*(1-filter_rate):.1f}%)")
    print(f"  filter_for_training   : {total_filter:,} ({100*filter_rate:.1f}%)")
    print(f"    of which parse_fail : {total_parse_fail:,} ({100*parse_rate:.2f}%)")
    print(f"  NaN/Inf files (spot)  : {nan_count}")
    print()
    print(f"  Illusion team replays : {total_illusion_t}")
    print(f"  Illusion break replays: {total_illusion_b}")
    print(f"  Wall time             : {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if total_processed > 0:
        print(f"  Throughput            : {total_processed/elapsed:.2f} replays/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
