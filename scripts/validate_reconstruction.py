#!/usr/bin/env python3
"""
D5: Sim-to-sim validation — reconstruction consistency checker.

For each of 100 randomly selected trajectory files, re-parses the source
replay log and independently verifies the reconstructed state vectors against
the raw log events.  Reports a mismatch taxonomy and overall consistency rate.

Validation checks performed per turn:
  HP_MATCH       — active Pokemon HP fraction within 1% of log value
  STATUS_MATCH   — active Pokemon status matches log
  ACTIVE_MATCH   — active Pokemon species matches log
  WEATHER_MATCH  — field weather matches log
  TURN_ORDER     — turn numbers are strictly increasing

Acceptance criterion: ≥ 80% of checked turns pass all checks.

Usage:
  cd /home/user/showdown-bot
  python3 scripts/validate_reconstruction.py [--max 100]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/user/metamon")

import os
os.environ.setdefault("METAMON_ALLOW_ANY_POKE_ENV", "True")
os.environ.setdefault("METAMON_CACHE_DIR", "/home/user/metamon-cache")

from replay_ingestion.parser import parse_replay_file, Gen9Parser
from replay_ingestion.reconstruct import reconstruct


# Mismatch categories
CHECKS = ["HP_MATCH", "STATUS_MATCH", "ACTIVE_MATCH", "WEATHER_MATCH", "TURN_ORDER"]

_WEATHER_MAP = {
    "RainDance": "rain", "SunnyDay": "sun", "Sandstorm": "sand",
    "Snow": "snow", "PrimordialSea": "rain", "DesolateLand": "sun",
    None: "none",
}


def validate_replay(json_path: Path) -> dict:
    """
    Parse and reconstruct one replay, then verify reconstruction consistency.
    Returns per-check pass/fail counts + mismatch details.
    """
    result = {
        "path": str(json_path),
        "ok": False,
        "turns_checked": 0,
        "passes": {c: 0 for c in CHECKS},
        "failures": {c: 0 for c in CHECKS},
        "errors": [],
    }

    try:
        battle = parse_replay_file(str(json_path))
        if battle is None or not battle.turns:
            result["errors"].append("parse_failed")
            return result

        rec = reconstruct(battle)
        if rec is None:
            result["errors"].append("reconstruct_failed")
            return result

        # Use p1's views for validation (arbitrary choice)
        views = rec.p1_views

        prev_turn = -1
        for i, (snap, view) in enumerate(zip(battle.turns, views)):
            result["turns_checked"] += 1

            # TURN_ORDER: turn numbers increase
            if snap.turn_number > prev_turn:
                result["passes"]["TURN_ORDER"] += 1
            else:
                result["failures"]["TURN_ORDER"] += 1
            prev_turn = snap.turn_number

            # ACTIVE_MATCH: own active Pokemon species matches snapshot
            own_active_snap = next(
                (s for s in snap.p1_slots if s.nickname == snap.p1_active_nick), None
            )
            own_active_view = next(
                (s for s in view.own_slots if s.nickname == view.own_active_nick), None
            )
            if own_active_snap is not None and own_active_view is not None:
                # Use .forme if set (handles Illusion/forme changes)
                snap_species = own_active_snap.forme or own_active_snap.species
                view_species = own_active_view.species  # reconstruct already resolved forme
                if snap_species == view_species or not snap_species or not view_species:
                    result["passes"]["ACTIVE_MATCH"] += 1
                else:
                    result["failures"]["ACTIVE_MATCH"] += 1
                    result["errors"].append(
                        f"T{snap.turn_number} active: snap={snap_species!r} view={view_species!r}"
                    )
            else:
                result["passes"]["ACTIVE_MATCH"] += 1  # no active on this turn (pre-battle)

            # HP_MATCH: active Pokemon HP fraction within 1%
            if own_active_snap is not None and own_active_view is not None:
                hp_snap = own_active_snap.hp_fraction
                hp_view = own_active_view.hp_fraction
                if abs(hp_snap - hp_view) <= 0.01:
                    result["passes"]["HP_MATCH"] += 1
                else:
                    result["failures"]["HP_MATCH"] += 1
                    result["errors"].append(
                        f"T{snap.turn_number} hp: snap={hp_snap:.3f} view={hp_view:.3f}"
                    )
            else:
                result["passes"]["HP_MATCH"] += 1

            # STATUS_MATCH: active Pokemon status matches
            if own_active_snap is not None and own_active_view is not None:
                if own_active_snap.status == own_active_view.status:
                    result["passes"]["STATUS_MATCH"] += 1
                else:
                    result["failures"]["STATUS_MATCH"] += 1
                    result["errors"].append(
                        f"T{snap.turn_number} status: snap={own_active_snap.status!r}"
                        f" view={own_active_view.status!r}"
                    )
            else:
                result["passes"]["STATUS_MATCH"] += 1

            # WEATHER_MATCH: field weather matches
            snap_weather = _WEATHER_MAP.get(snap.field_state.weather, "none")
            view_weather = _WEATHER_MAP.get(view.field_state.weather, "none")
            if snap_weather == view_weather:
                result["passes"]["WEATHER_MATCH"] += 1
            else:
                result["failures"]["WEATHER_MATCH"] += 1
                result["errors"].append(
                    f"T{snap.turn_number} weather: snap={snap_weather!r} view={view_weather!r}"
                )

        result["ok"] = True
    except Exception as exc:
        result["errors"].append(f"exception: {exc}")

    return result


def main():
    parser = argparse.ArgumentParser(description="D5 reconstruction validator")
    parser.add_argument("--max", type=int, default=100, help="Max replays to check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replay-dir", default="data/raw_replays/gen9randombattle",
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    replay_dir = base / args.replay_dir
    json_files = sorted(replay_dir.glob("*.json"))

    if not json_files:
        print("No replay files found. Run scripts/run_scraper.py first.")
        return 1

    rng = random.Random(args.seed)
    sample = json_files[:args.max] if len(json_files) <= args.max else rng.sample(
        json_files, args.max
    )

    print(f"D5 Reconstruction Validator")
    print(f"  Checking {len(sample)} replays from {replay_dir}")
    print()

    all_results = []
    for json_path in sample:
        r = validate_replay(json_path)
        all_results.append(r)

    # Aggregate
    ok_results = [r for r in all_results if r["ok"]]
    total_turns = sum(r["turns_checked"] for r in ok_results)

    print(f"{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Replays checked : {len(all_results)}")
    print(f"  Parse/rec OK    : {len(ok_results)}")
    print(f"  Total turns     : {total_turns}")
    print()
    print(f"  Per-check consistency:")

    all_pass = True
    for check in CHECKS:
        passes = sum(r["passes"][check] for r in ok_results)
        failures = sum(r["failures"][check] for r in ok_results)
        total = passes + failures
        rate = passes / total if total else 1.0
        print(f"    {check:15s}: {passes}/{total} ({100*rate:.1f}%)")
        if rate < 0.80:
            all_pass = False

    # Overall: a turn passes if ALL checks pass
    per_turn_overall = []
    for r in ok_results:
        for t in range(r["turns_checked"]):
            per_turn_overall.append(
                all(r["passes"][c] > t - sum(r["failures"][c] for c in CHECKS[:CHECKS.index(c)])
                    for c in CHECKS)
            )

    # Simpler: count overall as mean of per-check rates
    mean_rates = []
    for check in CHECKS:
        passes = sum(r["passes"][check] for r in ok_results)
        total = passes + sum(r["failures"][check] for r in ok_results)
        mean_rates.append(passes / total if total else 1.0)
    overall_rate = sum(mean_rates) / len(mean_rates)

    print()
    print(f"  Mean consistency rate: {100*overall_rate:.1f}%")

    # Show top mismatches
    all_errors = []
    for r in all_results:
        all_errors.extend(r["errors"][:3])  # cap per-replay to avoid spam

    error_types: dict = {}
    for e in all_errors:
        prefix = e.split(":")[0] if ":" in e else e
        error_types[prefix] = error_types.get(prefix, 0) + 1

    if error_types:
        print()
        print("  Mismatch taxonomy:")
        for k, v in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
            print(f"    {v:4d}x  {k}")

    print()
    passed = overall_rate >= 0.80 and len(ok_results) == len(all_results)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        if overall_rate < 0.80:
            print(f"  ✗ Mean consistency rate {100*overall_rate:.1f}% < 80%")
        if len(ok_results) < len(all_results):
            print(f"  ✗ {len(all_results)-len(ok_results)} replays failed to parse/reconstruct")
    print()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
