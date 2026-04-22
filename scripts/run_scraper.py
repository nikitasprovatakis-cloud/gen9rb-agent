#!/usr/bin/env python3
"""
D1 smoke-test: scrape up to N Gen 9 Random Battle replays at 1900+ rating.

Usage:
  cd /home/user/showdown-bot
  python scripts/run_scraper.py [--max 100] [--min-rating 1900]

Output directory: data/raw_replays/gen9randombattle/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from replay_ingestion.scraper import ReplayScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Scrape Gen 9 Random Battle replays")
    parser.add_argument("--max", type=int, default=100, help="Max replays to download")
    parser.add_argument("--min-rating", type=int, default=1900)
    parser.add_argument("--max-age-days", type=int, default=365)
    parser.add_argument(
        "--output-dir",
        default="data/raw_replays/gen9randombattle",
        help="Directory to save replays",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / args.output_dir
    scraper = ReplayScraper(
        output_dir=str(output_dir),
        min_rating=args.min_rating,
        max_age_days=args.max_age_days,
    )

    print(f"Scraping up to {args.max} Gen 9 Random Battle replays (rating ≥ {args.min_rating})")
    print(f"Output: {output_dir}")
    print()

    stats = scraper.scrape(fmt="gen9randombattle", max_replays=args.max)

    print()
    print("=" * 50)
    print("SCRAPE COMPLETE")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    report = scraper.report()
    print()
    print("Cached replay stats:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
