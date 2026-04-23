#!/usr/bin/env python3
"""
Step 4: Player supplement scraper for Phase 3 dataset.

Scrapes gen9randombattle replays from specific high-ELO players posted after
July 22, 2025 (the HolidayOugi dataset cutoff), applying 1900+ rating filter.

Usage:
  python scripts/run_player_supplement.py [--verify-only] [--dry-run]

  --verify-only  : Only run player account verification (Step 3), no scraping.
  --dry-run      : Print plan without downloading anything.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/user/metamon")

import os
os.environ.setdefault("METAMON_ALLOW_ANY_POKE_ENV", "True")
os.environ.setdefault("METAMON_CACHE_DIR", "/home/user/metamon-cache")

LOG_PATH = Path(__file__).parent.parent / "logs" / "player_supplement.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_player_supplement")

# ---------------------------------------------------------------------------
# Player list — display names and normalized Showdown IDs
# ---------------------------------------------------------------------------

RAW_PLAYERS = [
    "Aqua",
    "Michaelderbeste2",
    "pokeblade101",
    "teresbahji",
    "milkreo",
    "referrals",
    "smokyaim",
    "articoo",
    "delta2777",
    "dra15v2",
    "helicopyer",
    "70to90gxe",
    "masterj007",
    "sigurdzz",
    "drizzle",
    "pentav",
    "wintersim",
    "galak0",
    "cephaleid",
    "fatmarmot",
    "assidion",
    "daruma",
    "bauses",
    "firehills",
    "sebasdb",
    "norman2!",
    "lizardune",
    "emptybrackets",
    "mylifeisdance",
    "piyush21",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify player accounts, do not scrape")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without downloading")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    holidayougi_dir = base / "data" / "holidayougi"
    supplement_dir  = base / "data" / "raw_replays" / "supplement"
    traj_dir        = base / "data" / "trajectories" / "gen9randombattle"

    from replay_ingestion.player_scraper import PlayerScraper, normalize_id

    # ── Step 3 equivalent: normalize + verify ──────────────────────────────
    scraper = PlayerScraper(supplement_dir)

    players_norm = [(name, normalize_id(name)) for name in RAW_PLAYERS]

    print()
    print("=" * 65)
    print("PLAYER LIST NORMALIZATION")
    print("=" * 65)
    print(f"{'Display name':<25} {'Normalized ID':<25} Status")
    print("-" * 65)

    verified = []
    failed   = []
    for display, norm_id in players_norm:
        if args.dry_run:
            status = "DRY-RUN"
            verified.append((display, norm_id))
        else:
            ok = scraper.verify_player(norm_id)
            status = "OK" if ok else "FAIL — no gen9RB replays found"
            if ok:
                verified.append((display, norm_id))
            else:
                failed.append((display, norm_id))
        print(f"  {display:<23} {norm_id:<25} {status}")

    print()
    if failed:
        print(f"WARNING: {len(failed)} player(s) did not resolve:")
        for d, n in failed:
            print(f"  '{d}' → '{n}'")
        print("Please correct spelling and re-run.")
    else:
        print(f"All {len(verified)} players verified OK.")

    if args.verify_only or args.dry_run:
        return 0

    if failed:
        print("\nAborting: unresolved players. Fix and re-run.")
        return 1

    # ── Step 4: Load HolidayOugi IDs for dedup ────────────────────────────
    print()
    print("Loading HolidayOugi IDs for dedup check...")
    t0 = time.time()
    hids = PlayerScraper.load_holidayougi_ids(holidayougi_dir)
    logger.info("HolidayOugi ID set: %d IDs loaded in %.1fs", len(hids), time.time()-t0)

    scraper.holidayougi_ids = hids

    # ── Step 4: Scrape ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("PLAYER SUPPLEMENT SCRAPE")
    print("=" * 65)
    logger.info("Starting player supplement scrape. Players: %d", len(verified))

    t_start = time.time()
    agg = scraper.scrape_all(verified)
    elapsed = time.time() - t_start

    # ── Supplement summary ─────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("SUPPLEMENT SCRAPE SUMMARY")
    print("=" * 65)
    print(f"  Total unique replays  : {agg['total_unique_replays']:,}")
    print(f"  Shared (multi-player) : {agg['shared']:,}")
    print(f"  Skipped (rating)      : {agg['skipped_rating']:,}")
    print(f"  Skipped (date)        : {agg['skipped_date']:,}")
    print(f"  Skipped (exists)      : {agg['skipped_exists']:,}")
    print(f"  Skipped (holidayougi) : {agg['skipped_holidayougi']:,}")
    print(f"  Errors                : {agg['errors']:,}")
    print(f"  Wall time             : {elapsed:.1f}s")
    print()
    print("Per-player breakdown:")
    print(f"  {'Player':<25} {'Downloaded':>10} {'Shared':>7} {'Skip/rate':>10} {'Errors':>7}")
    print("  " + "-" * 60)
    for display, stats in agg["per_player"].items():
        print(f"  {display:<25} {stats['downloaded']:>10,} {stats['shared']:>7,} "
              f"{stats['skipped_rating']:>10,} {stats['errors']:>7,}")

    # Rating + date stats for supplement
    import json as _json
    supplement_files = sorted(supplement_dir.glob("*.json"))
    ratings, timestamps = [], []
    for f in supplement_files:
        try:
            data = _json.loads(f.read_text())
            r = data.get("rating")
            t = data.get("uploadtime")
            if r:
                ratings.append(float(r))
            if t:
                timestamps.append(int(t))
        except Exception:
            pass

    if ratings:
        import statistics
        print()
        print(f"  Rating distribution ({len(ratings)} rated):")
        print(f"    Min: {min(ratings):.0f}  Max: {max(ratings):.0f}  "
              f"Mean: {statistics.mean(ratings):.1f}  "
              f"Median: {statistics.median(ratings):.1f}")
    if timestamps:
        from datetime import datetime, timezone
        ts_to_date = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d')
        print(f"  Date range: {ts_to_date(min(timestamps))} – {ts_to_date(max(timestamps))}")

        # Save cutoff to DECISIONS.md
        cutoff_ts = max(timestamps)
        cutoff_dt = ts_to_date(cutoff_ts)
        decisions_path = Path(__file__).parent.parent / "DECISIONS.md"
        if decisions_path.exists():
            entry = (
                f"\n\n## Main Dataset Collection Cutoff\n\n"
                f"Player supplement scrape completed: {cutoff_dt}\n"
                f"Latest replay uploadtime included: {cutoff_ts} ({cutoff_dt} UTC)\n"
                f"Phase 9 data collection must use uploadtime > {cutoff_ts} "
                f"to avoid overlap.\n"
            )
            with open(decisions_path, "a") as fh:
                fh.write(entry)
            logger.info("Cutoff written to DECISIONS.md: %d (%s)", cutoff_ts, cutoff_dt)

    return 0


if __name__ == "__main__":
    sys.exit(main())
