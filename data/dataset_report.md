# Dataset Report — Gen 9 Random Battle

Generated: 2026-04-22 20:24 UTC

## Replay Collection (D1)

| Metric | Value |
|--------|-------|
| Replays scraped | 100 |
| Rating range | 1901 – 2343 |
| Rating mean | 2140 |
| Date range | 2026-04-22 to 2026-04-22 |
| Min rating filter | 1900 |
| Max age filter | 365 days |

## Trajectory Dataset (D4)

| Metric | Value |
|--------|-------|
| Total trajectory files (.npz) | 200 |
| P1 POV files | 100 |
| P2 POV files | 100 |
| Total turns (both POVs) | 5,450 |
| Avg turns per POV | 27.2 |
| Valid action labels (≥0) | 5,244 (96.2%) |
| Invalid action labels (-1) | 206 (3.8%) |
| Files with NaN/Inf | 0 |
| Avg legal actions per turn | 6.9 / 13 |
| Feature vector dimension | 959 |

## Action Distribution

Based on 5,244 valid action labels.

| Slot | Type | Count | % |
|------|------|-------|---|
| 0 | Move 0 (alphabetical rank 1) | 1,868 | 35.6% |
| 1 | Move 1 (alphabetical rank 2) | 1,195 | 22.8% |
| 2 | Move 2 (alphabetical rank 3) | 644 | 12.3% |
| 3 | Move 3 (alphabetical rank 4) | 220 | 4.2% |
| 4 | Switch 0 (alphabetical rank 1) | 404 | 7.7% |
| 5 | Switch 1 (alphabetical rank 2) | 296 | 5.6% |
| 6 | Switch 2 (alphabetical rank 3) | 256 | 4.9% |
| 7 | Switch 3 (alphabetical rank 4) | 147 | 2.8% |
| 8 | Switch 4 (alphabetical rank 5) | 77 | 1.5% |
| 9 | Tera + Move 0 | 71 | 1.4% |
| 10 | Tera + Move 1 | 38 | 0.7% |
| 11 | Tera + Move 2 | 26 | 0.5% |
| 12 | Tera + Move 3 | 2 | 0.0% |

## Winner Distribution

| Winner | POV Files |
|--------|-----------|
| Player 1 wins | 114 |
| Player 2 wins | 86 |
| Unknown/tie | 0 |

## Notes

- **Action = -1**: Occurs when the parser could not record a clean action.
  Main causes: (1) `cant` events (sleep/paralysis prevented move choice),
  (2) force-switch flag mis-timing (switch happened previous turn),
  (3) last turn of game (no action after `|win|`).
  These turns are excluded from policy loss computation during training.
- **Feature [0]**: species index for `nn.Embedding(509, embed_dim)` — not a
  continuous float. Phase 4 must handle separately from the 76 continuous features.
- **Damage calc**: Feature [55] uses full DamageCalculator. Systematic 2–4 HP
  overestimate vs Showdown due to rng-ordering difference (Metamon formula).
  Accepted limitation; not expected to affect Phase 8 MCTS decisions materially.
