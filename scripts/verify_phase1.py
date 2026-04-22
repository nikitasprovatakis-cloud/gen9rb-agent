#!/usr/bin/env python3
"""
Sanity check: confirms the Phase 1 stack is alive.
Runs 10 MaxDamageBot vs RandomPlayer battles and asserts win rate >= 70%.
Requires: Showdown server on localhost:8000, metamon-env activated.
"""

import os
import sys
import asyncio

BASE_DIR = "/home/user"
METAMON_DIR = f"{BASE_DIR}/metamon"

os.environ["METAMON_ALLOW_ANY_POKE_ENV"] = "True"
os.environ["METAMON_CACHE_DIR"] = f"{BASE_DIR}/metamon-cache"
if METAMON_DIR not in sys.path:
    sys.path.insert(0, METAMON_DIR)

import warnings
warnings.filterwarnings("ignore")


def _check_server():
    import urllib.request
    try:
        code = urllib.request.urlopen("http://localhost:8000", timeout=3).getcode()
        assert code == 200, f"HTTP {code}"
    except Exception as e:
        print(f"FAIL: Showdown server not reachable at localhost:8000 — {e}")
        print("      Start it with: node pokemon-showdown start --no-security")
        sys.exit(1)
    print("  [OK] Showdown server reachable")


async def _run(n: int) -> float:
    from poke_env import LocalhostServerConfiguration
    from poke_env.player import RandomPlayer
    from poke_env.data import to_id_str
    sys.path.insert(0, f"{BASE_DIR}/showdown-bot")
    from max_damage_bot import MaxDamageBot

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
    await asyncio.gather(
        bot.send_challenges(to_id_str(rand.username), n_challenges=n,
                            to_wait=rand.ps_client.logged_in),
        rand.accept_challenges(to_id_str(bot.username), n_challenges=n,
                               packed_team=rand.next_team),
    )
    return bot.win_rate


def main():
    print("=" * 50)
    print("Phase 1 stack verification")
    print("=" * 50)

    _check_server()

    n = 10
    print(f"  Running {n} MaxDamageBot vs RandomPlayer battles...")
    win_rate = asyncio.get_event_loop().run_until_complete(_run(n))
    wins = round(win_rate * n)
    pct = round(win_rate * 100, 1)
    print(f"  Result: {wins}/{n} wins ({pct}%)")

    if win_rate >= 0.70:
        print(f"\nPHASE 1 OK — MaxDamageBot {wins}/{n} wins vs RandomPlayer")
    else:
        print(f"\nFAIL — win rate {pct}% below 70% threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
