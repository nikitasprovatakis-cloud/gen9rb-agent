#!/usr/bin/env python3
"""
Phase 1 Deliverable 4: MaxDamageBot

A Gen 9 Random Battle heuristic bot using poke-env.

Attack logic:
  - Score each available move by: base_power * STAB * type_effectiveness
  - Pick the highest-scoring move; break STAB ties in favour of STAB moves
  - Never voluntarily Tera

Switch logic (force_switch only):
  - Score each available switch by: incoming type disadvantage sum (lower = better)
  - Break ties by highest remaining HP fraction

Edge cases handled via poke-env's available_moves / available_switches:
  - Choice-locked / Encore-locked moves: poke-env returns only the legal move
  - Struggle: available_moves=[<struggle>] when PP is zero on all moves
  - Trapped: available_switches is empty when trapped
  - All paths fall back to choose_random_move() when nothing else applies
"""

import os
import sys
import time
import asyncio
import logging
import json
import datetime

# Must be set before importing metamon
os.environ["METAMON_ALLOW_ANY_POKE_ENV"] = "True"
os.environ["METAMON_CACHE_DIR"] = "/home/user/metamon-cache"
sys.path.insert(0, "/home/user/metamon")

import warnings
warnings.filterwarnings("ignore")

from poke_env.player import Player, RandomPlayer
from poke_env.environment import Move, Pokemon, Battle, MoveCategory

LOG_DIR = "/home/user/showdown-bot/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("MaxDamageBot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _type_effectiveness(move: Move, target: Pokemon) -> float:
    """Product of move type against target's type(s) using poke-env's damage_multiplier."""
    return target.damage_multiplier(move)


def _stab_mult(move: Move, user: Pokemon) -> float:
    return 1.5 if move.type in user.types else 1.0


def _move_score(move: Move, user: Pokemon, target: Pokemon) -> float:
    """Heuristic: base_power × STAB × type_effectiveness."""
    bp = move.base_power or 0
    if bp == 0:
        return 0.0
    return bp * _stab_mult(move, user) * _type_effectiveness(move, target)


def _best_attack(available_moves, user: Pokemon, target: Pokemon):
    """Pick the highest-scoring move; ties broken by STAB preference."""
    if not available_moves:
        return None
    scored = [(_move_score(m, user, target), _stab_mult(m, user), m) for m in available_moves]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def _switch_score(switch: Pokemon, opponent: Pokemon) -> float:
    """Lower = better: sum of opponent's type effectiveness against the switch-in."""
    total = 0.0
    for opp_type in (t for t in opponent.types if t is not None):
        total += switch.damage_multiplier(opp_type)
    return total


def _best_switch(available_switches, opponent: Pokemon):
    """Switch with lowest incoming type effectiveness; ties broken by highest HP %."""
    if not available_switches:
        return None
    scored = [
        (_switch_score(sw, opponent), -sw.current_hp_fraction, sw)
        for sw in available_switches
    ]
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


class MaxDamageBot(Player):
    """
    Heuristic Gen 9 Random Battle bot.
    Never voluntarily switches; never uses Tera.
    """

    def choose_move(self, battle: Battle):
        user = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # ── Force-switch path ──────────────────────────────────────────────
        if battle.force_switch:
            if battle.available_switches:
                best = _best_switch(battle.available_switches, opponent)
                return self.create_order(best)
            return self.choose_random_move(battle)

        # ── Attack path ────────────────────────────────────────────────────
        if battle.available_moves:
            best = _best_attack(battle.available_moves, user, opponent)
            if best is not None:
                return self.create_order(best)

        # ── Fallback (Struggle, misc edge cases) ──────────────────────────
        return self.choose_random_move(battle)


# ─── Evaluation helpers ───────────────────────────────────────────────────────

class _BattleLogger:
    """Accumulates per-battle results and writes a JSONL log file."""

    def __init__(self, path: str):
        self.path = path
        self.battles: list[dict] = []
        self._start = time.time()

    def record(self, battle_tag: str, won: bool, turns: int, opponent: str, fmt: str):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "format": fmt,
            "opponent": opponent,
            "result": "WIN" if won else "LOSS",
            "turns": turns,
            "replay_url": None,
        }
        self.battles.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def summary(self) -> dict:
        wins = sum(1 for b in self.battles if b["result"] == "WIN")
        return {
            "total": len(self.battles),
            "wins": wins,
            "losses": len(self.battles) - wins,
            "win_pct": round(100 * wins / max(len(self.battles), 1), 1),
            "elapsed_s": round(time.time() - self._start, 1),
        }


async def _run_battles_vs_random(n: int, logger_obj: _BattleLogger, per_battle_timeout: float = 300):
    """Run n battles between MaxDamageBot and RandomPlayer using direct challenges."""
    from poke_env import LocalhostServerConfiguration
    from poke_env.player import RandomPlayer
    from poke_env.data import to_id_str

    bot = MaxDamageBot(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )
    rand = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    # Use direct challenge so bot.battles is populated (cross_evaluate calls reset_battles())
    await asyncio.gather(
        bot.send_challenges(
            opponent=to_id_str(rand.username),
            n_challenges=n,
            to_wait=rand.ps_client.logged_in,
        ),
        rand.accept_challenges(
            opponent=to_id_str(bot.username),
            n_challenges=n,
            packed_team=rand.next_team,
        ),
    )

    for battle in bot.battles.values():
        logger_obj.record(
            battle_tag=battle.battle_tag,
            won=battle.won,
            turns=battle.turn,
            opponent=battle.opponent_username or "RandomPlayer",
            fmt="gen9randombattle",
        )

    return logger_obj.summary()


async def _run_battles_vs_metamon(n: int, logger_obj: _BattleLogger):
    """
    Run n battles: MaxDamageBot (challenger) vs MetamonBot (accepter).
    MetamonBot is Metamon's SmallRLGen9Beta checkpoint running on CPU.
    """
    import metamon.config
    metamon.config.FORMAT_ALIASES["gen9randombattle"] = "gen9ou"
    from metamon.rl.pretrained import SmallRLGen9Beta
    from poke_env import LocalhostServerConfiguration

    model = SmallRLGen9Beta()
    model.gin_overrides = None  # use VanillaAttention fallback (CPU safe)

    # MaxDamageBot directly challenges
    bot = MaxDamageBot(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    # We need the Metamon agent to be available and accept challenges.
    # The simplest approach for Phase 1: spawn Metamon via QueueOnLocalLadder
    # in a subprocess, then have our bot challenge it.
    # Since cross_evaluate requires two poke-env Players, we use a simpler challenge approach.

    # Run Metamon as a ladder queue in a thread, then challenge it
    import threading

    metamon_exception: list[Exception] = []

    def _run_metamon():
        try:
            from metamon.rl.evaluate import pretrained_vs_local_ladder
            pretrained_vs_local_ladder(
                pretrained_model=model,
                username="MetamonOpponent",
                battle_format="gen9randombattle",
                team_set=None,
                total_battles=n,
                battle_backend="pokeagent",
                log_to_wandb=False,
            )
        except Exception as e:
            metamon_exception.append(e)

    metamon_thread = threading.Thread(target=_run_metamon, daemon=True)
    metamon_thread.start()
    await asyncio.sleep(15)  # give Metamon time to load and connect

    # Challenge Metamon via ladder (both queueing for same format)
    await bot.ladder(n)
    metamon_thread.join(timeout=60)

    if metamon_exception:
        raise metamon_exception[0]

    for battle in bot.battles.values():
        logger_obj.record(
            battle_tag=battle.battle_tag,
            won=battle.won,
            turns=battle.turn,
            opponent="MetamonSmallRLGen9Beta",
            fmt="gen9randombattle",
        )

    return logger_obj.summary()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run MaxDamageBot evaluation")
    p.add_argument("--vs-random", type=int, default=100, help="Battles vs RandomPlayer")
    p.add_argument("--vs-metamon", type=int, default=0, help="Battles vs Metamon checkpoint")
    p.add_argument("--log", default=f"{LOG_DIR}/maxdamagebot.jsonl", help="Log file path")
    args = p.parse_args()

    battle_logger = _BattleLogger(args.log)

    print("=" * 60)
    print(f"MaxDamageBot Evaluation")
    print("=" * 60)

    if args.vs_random > 0:
        print(f"\nRunning {args.vs_random} battles vs RandomPlayer...")
        summary = asyncio.get_event_loop().run_until_complete(
            _run_battles_vs_random(args.vs_random, battle_logger)
        )
        wins = summary["wins"]
        total = summary["total"]
        pct = summary["win_pct"]
        print(f"  Result: {wins}/{total} wins ({pct}%) in {summary['elapsed_s']}s")
        if pct < 70:
            print(f"  WARN: Win rate {pct}% < 70% target — check bot logic")
        else:
            print(f"  PASS: Win rate meets ≥70% target")

    if args.vs_metamon > 0:
        print(f"\nRunning {args.vs_metamon} battles vs Metamon SmallRLGen9Beta...")
        try:
            summary = asyncio.get_event_loop().run_until_complete(
                _run_battles_vs_metamon(args.vs_metamon, battle_logger)
            )
            print(f"  Completed without crash. Win rate: {summary['win_pct']}%")
            print(f"  PASS: No crashes in {args.vs_metamon} battles vs Metamon")
        except Exception as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

    print(f"\nLog: {args.log}")
    print(f"Total battles logged: {len(battle_logger.battles)}")


if __name__ == "__main__":
    main()
