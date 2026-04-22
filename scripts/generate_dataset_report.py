#!/usr/bin/env python3
"""
D6: Dataset summary report generator.

Scans data/trajectories/gen9randombattle/*.npz and
data/raw_replays/gen9randombattle/*.json to produce a markdown
report at data/dataset_report.md.

Usage:
  cd /home/user/showdown-bot
  python3 scripts/generate_dataset_report.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/user/metamon")

import os
os.environ.setdefault("METAMON_ALLOW_ANY_POKE_ENV", "True")
os.environ.setdefault("METAMON_CACHE_DIR", "/home/user/metamon-cache")


def main():
    base = Path(__file__).parent.parent
    replay_dir = base / "data/raw_replays/gen9randombattle"
    traj_dir   = base / "data/trajectories/gen9randombattle"
    report_path = base / "data/dataset_report.md"

    from knowledge.features import FEATURE_DIM

    # ── Replay stats ──────────────────────────────────────────────────────────
    json_files = sorted(replay_dir.glob("*.json"))
    ratings = []
    upload_times = []
    for jf in json_files:
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            r = d.get("rating")
            if isinstance(r, (int, float)):
                ratings.append(int(r))
            t = d.get("uploadtime")
            if t:
                upload_times.append(t)
        except Exception:
            pass

    # ── Trajectory stats ─────────────────────────────────────────────────────
    npz_files = sorted(traj_dir.glob("*.npz"))
    p1_files = [f for f in npz_files if "_p1" in f.name]
    p2_files = [f for f in npz_files if "_p2" in f.name]

    total_turns = 0
    valid_action_turns = 0
    nan_files = 0
    winner_counts = {1: 0, 2: 0, -1: 0}
    action_dist = {}
    legal_bits = []

    for f in npz_files:
        try:
            d = np.load(f)
            acts = d["actions"]
            states = d["states"]
            masks = d["legal_masks"]
            winner = int(d["winner"][0])

            total_turns += len(acts)
            valid_action_turns += int((acts >= 0).sum())
            legal_bits.append(masks.sum(axis=1).mean())
            winner_counts[winner] = winner_counts.get(winner, 0) + 1

            for a in acts[acts >= 0]:
                action_dist[int(a)] = action_dist.get(int(a), 0) + 1

            if np.any(np.isnan(states)) or np.any(np.isinf(states)):
                nan_files += 1
        except Exception:
            pass

    total_valid_acts = sum(action_dist.values())
    avg_legal = float(np.mean(legal_bits)) if legal_bits else 0.0

    # ── Date range ────────────────────────────────────────────────────────────
    if upload_times:
        ts_min = datetime.utcfromtimestamp(min(upload_times)).strftime("%Y-%m-%d")
        ts_max = datetime.utcfromtimestamp(max(upload_times)).strftime("%Y-%m-%d")
    else:
        ts_min = ts_max = "unknown"

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Build report ─────────────────────────────────────────────────────────
    lines = [
        f"# Dataset Report — Gen 9 Random Battle",
        f"",
        f"Generated: {now}",
        f"",
        f"## Replay Collection (D1)",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Replays scraped | {len(json_files)} |",
        f"| Rating range | {min(ratings) if ratings else 'N/A'} – {max(ratings) if ratings else 'N/A'} |",
        f"| Rating mean | {int(sum(ratings)/len(ratings)) if ratings else 'N/A'} |",
        f"| Date range | {ts_min} to {ts_max} |",
        f"| Min rating filter | 1900 |",
        f"| Max age filter | 365 days |",
        f"",
        f"## Trajectory Dataset (D4)",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total trajectory files (.npz) | {len(npz_files)} |",
        f"| P1 POV files | {len(p1_files)} |",
        f"| P2 POV files | {len(p2_files)} |",
        f"| Total turns (both POVs) | {total_turns:,} |",
        f"| Avg turns per POV | {total_turns/len(npz_files):.1f} |",
        f"| Valid action labels (≥0) | {valid_action_turns:,} ({100*valid_action_turns/total_turns:.1f}%) |",
        f"| Invalid action labels (-1) | {total_turns-valid_action_turns:,} ({100*(total_turns-valid_action_turns)/total_turns:.1f}%) |",
        f"| Files with NaN/Inf | {nan_files} |",
        f"| Avg legal actions per turn | {avg_legal:.1f} / 13 |",
        f"| Feature vector dimension | {FEATURE_DIM} |",
        f"",
        f"## Action Distribution",
        f"",
        f"Based on {total_valid_acts:,} valid action labels.",
        f"",
        f"| Slot | Type | Count | % |",
        f"|------|------|-------|---|",
    ]

    slot_labels = {
        0: "Move 0 (alphabetical rank 1)",
        1: "Move 1 (alphabetical rank 2)",
        2: "Move 2 (alphabetical rank 3)",
        3: "Move 3 (alphabetical rank 4)",
        4: "Switch 0 (alphabetical rank 1)",
        5: "Switch 1 (alphabetical rank 2)",
        6: "Switch 2 (alphabetical rank 3)",
        7: "Switch 3 (alphabetical rank 4)",
        8: "Switch 4 (alphabetical rank 5)",
        9: "Tera + Move 0",
        10: "Tera + Move 1",
        11: "Tera + Move 2",
        12: "Tera + Move 3",
    }

    for slot in range(13):
        cnt = action_dist.get(slot, 0)
        pct = 100 * cnt / total_valid_acts if total_valid_acts else 0
        lines.append(f"| {slot} | {slot_labels.get(slot, '?')} | {cnt:,} | {pct:.1f}% |")

    lines += [
        f"",
        f"## Winner Distribution",
        f"",
        f"| Winner | POV Files |",
        f"|--------|-----------|",
        f"| Player 1 wins | {winner_counts.get(1, 0)} |",
        f"| Player 2 wins | {winner_counts.get(2, 0)} |",
        f"| Unknown/tie | {winner_counts.get(-1, 0)} |",
        f"",
        f"## Notes",
        f"",
        f"- **Action = -1**: Occurs when the parser could not record a clean action.",
        f"  Main causes: (1) `cant` events (sleep/paralysis prevented move choice),",
        f"  (2) force-switch flag mis-timing (switch happened previous turn),",
        f"  (3) last turn of game (no action after `|win|`).",
        f"  These turns are excluded from policy loss computation during training.",
        f"- **Feature [0]**: species index for `nn.Embedding(509, embed_dim)` — not a",
        f"  continuous float. Phase 4 must handle separately from the 76 continuous features.",
        f"- **Damage calc**: Feature [55] uses full DamageCalculator. Systematic 2–4 HP",
        f"  overestimate vs Showdown due to rng-ordering difference (Metamon formula).",
        f"  Accepted limitation; not expected to affect Phase 8 MCTS decisions materially.",
        f"",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    for line in lines[:50]:
        print(line)


if __name__ == "__main__":
    main()
