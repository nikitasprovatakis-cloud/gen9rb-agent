#!/usr/bin/env python3
"""
Phase 2 D6: Integration test — knowledge layer end-to-end.

Runs 10 live Gen 9 Random Battle games against RandomPlayer.
On every turn, extracts a feature vector using BattleFeatureExtractor and
prints a compact diagnostic:
  - Turn number, own active Pokemon, opponent active Pokemon
  - Top-2 predicted roles for the opponent with probabilities
  - Key feature values: HP fraction, status, type coverage, item probs

Acceptance criteria:
  - 10 battles complete without crashes
  - Feature vector shape = (947,) on every turn
  - No NaN or Inf values
  - SetPredictor updates after observing moves

Usage:
  cd /home/user/showdown-bot
  python scripts/integration_test_phase2.py
"""

import asyncio
import logging
import os
import sys
import time

os.environ["METAMON_ALLOW_ANY_POKE_ENV"] = "True"
os.environ["METAMON_CACHE_DIR"] = "/home/user/metamon-cache"
sys.path.insert(0, "/home/user/metamon")
sys.path.insert(0, "/home/user/showdown-bot")

import numpy as np

from poke_env.player import Player, RandomPlayer
from poke_env import LocalhostServerConfiguration

from knowledge.features import BattleFeatureExtractor, FEATURE_DIM
from knowledge.set_pool import to_id

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("integration_test")


# ── Diagnostic printer ───────────────────────────────────────────────────────

def _print_turn_diagnostic(battle, extractor: BattleFeatureExtractor, vec: np.ndarray) -> None:
    """Print a compact per-turn summary."""
    turn = battle.turn
    own = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    own_name = own.species if own else "???"
    own_hp = f"{own.current_hp_fraction*100:.0f}%" if own else "???"
    opp_name = opp.species if opp else "???"
    opp_hp = f"{opp.current_hp_fraction*100:.0f}%" if opp else "???"

    print(f"  T{turn:02d} | {own_name} ({own_hp}) vs {opp_name} ({opp_hp})")

    # Top-2 predicted roles for opponent
    if opp and opp_name != "???":
        sid = to_id(opp_name)
        pred = extractor._opp_predictors.get(sid)
        if pred is not None:
            dist = pred.get_distribution()[:2]
            parts = ", ".join(f"{role}={prob:.1%}" for role, _, prob in dist)
            print(f"         opp roles: [{parts}]")

    # Key feature values from the vector
    # Own active slot is slot 0 (sorted: active first)
    # We report: hp (idx 1), item probs (idx 46-53), expected damage norm (idx 54)
    slot0_start = 0
    hp_feat = vec[slot0_start + 1]
    item_probs = vec[slot0_start + 46: slot0_start + 54]
    dmg_norm = vec[slot0_start + 54]
    print(f"         own[0]: hp_feat={hp_feat:.2f}, best_move_dmg={dmg_norm*200:.0f}bp-equiv")

    # Vector validity
    has_nan = np.any(np.isnan(vec))
    has_inf = np.any(np.isinf(vec))
    if has_nan or has_inf:
        print(f"  *** WARN: NaN={has_nan}, Inf={has_inf} in feature vector ***")


# ── Diagnostic player ────────────────────────────────────────────────────────

class DiagnosticPlayer(Player):
    """
    Plays random moves but hooks into choose_move to run feature extraction
    and print diagnostics on every turn.
    """

    def __init__(self, extractor: BattleFeatureExtractor, **kwargs):
        super().__init__(**kwargs)
        self._extractor = extractor
        self._battle_count = 0
        self._turn_stats: list[dict] = []  # per-turn timing/shape records

    def choose_move(self, battle):
        # Reset extractor at battle start (first turn = turn 1)
        if battle.turn == 1 and battle.battle_tag not in self._seen_battles:
            self._seen_battles.add(battle.battle_tag)
            self._extractor.reset()
            self._battle_count += 1
            print(f"\n{'='*60}")
            print(f"Battle {self._battle_count}/10  [{battle.battle_tag}]")
            print(f"{'='*60}")

        t0 = time.perf_counter()
        vec = self._extractor.extract(battle)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._turn_stats.append({
            "battle": battle.battle_tag,
            "turn": battle.turn,
            "shape_ok": vec.shape == (FEATURE_DIM,),
            "nan": bool(np.any(np.isnan(vec))),
            "inf": bool(np.any(np.isinf(vec))),
            "elapsed_ms": elapsed_ms,
        })

        _print_turn_diagnostic(battle, self._extractor, vec)

        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        result = "WIN" if battle.won else "LOSS"
        print(f"  → {result} in {battle.turn} turns")

    def reset_battles(self):
        super().reset_battles()
        self._seen_battles = set()

    # Override to initialize _seen_battles before battles start
    def __init__(self, extractor: BattleFeatureExtractor, **kwargs):
        super().__init__(**kwargs)
        self._extractor = extractor
        self._battle_count = 0
        self._turn_stats: list[dict] = []
        self._seen_battles: set[str] = set()


# ── Main runner ───────────────────────────────────────────────────────────────

async def run_integration_test(n_battles: int = 10):
    from poke_env.data import to_id_str

    print("Phase 2 Knowledge Layer — Integration Test")
    print(f"Running {n_battles} battles vs RandomPlayer on localhost:8000")
    print(f"Feature vector dimension: {FEATURE_DIM}")
    print()

    extractor = BattleFeatureExtractor()
    print(f"Loaded species list: {len(extractor._species_list)-1} species")
    print(f"Loaded move DB: {len(extractor._move_db)} moves")
    print()

    bot = DiagnosticPlayer(
        extractor=extractor,
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )
    rand = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    t_start = time.time()

    await asyncio.gather(
        bot.send_challenges(
            opponent=to_id_str(rand.username),
            n_challenges=n_battles,
            to_wait=rand.ps_client.logged_in,
        ),
        rand.accept_challenges(
            opponent=to_id_str(bot.username),
            n_challenges=n_battles,
            packed_team=rand.next_team,
        ),
    )

    elapsed = time.time() - t_start

    # ── Final report ─────────────────────────────────────────────────────────
    stats = bot._turn_stats
    total_turns = len(stats)
    battles_done = len(bot.battles)
    wins = sum(1 for b in bot.battles.values() if b.won)

    shape_failures = [s for s in stats if not s["shape_ok"]]
    nan_turns = [s for s in stats if s["nan"]]
    inf_turns = [s for s in stats if s["inf"]]
    slow_turns = [s for s in stats if s["elapsed_ms"] > 50.0]

    timings = [s["elapsed_ms"] for s in stats]
    mean_ms = sum(timings) / len(timings) if timings else 0.0
    max_ms = max(timings) if timings else 0.0

    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Battles: {battles_done}/{n_battles}  ({wins}W / {battles_done-wins}L)")
    print(f"  Turns processed: {total_turns}")
    print(f"  Wall time: {elapsed:.1f}s")
    print()
    print(f"  Feature vector shape (947) OK: {total_turns - len(shape_failures)}/{total_turns}")
    print(f"  NaN turns:    {len(nan_turns)}")
    print(f"  Inf turns:    {len(inf_turns)}")
    print(f"  Slow turns (>50ms): {len(slow_turns)}")
    print(f"  Extraction timing: mean={mean_ms:.1f}ms, max={max_ms:.1f}ms")
    print()

    # Acceptance verdict
    passed = (
        battles_done == n_battles
        and len(shape_failures) == 0
        and len(nan_turns) == 0
        and len(inf_turns) == 0
    )
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        if battles_done < n_battles:
            print(f"    ✗ Only {battles_done}/{n_battles} battles completed")
        if shape_failures:
            print(f"    ✗ {len(shape_failures)} shape errors")
        if nan_turns:
            print(f"    ✗ NaN in {len(nan_turns)} turns")
        if inf_turns:
            print(f"    ✗ Inf in {len(inf_turns)} turns")
    print()

    return passed


if __name__ == "__main__":
    result = asyncio.run(run_integration_test(n_battles=10))
    sys.exit(0 if result else 1)
