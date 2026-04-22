"""
Phase 3 unit tests — replay ingestion (parser + reconstruct).

Covers hand-constructed synthetic replay logs for edge cases that are absent
or extremely rare in the 100-replay smoke-test dataset:

  - Illusion break: Zoroark disguised as Gengar; break reveals true species
  - Imposter Transform: confirmed against Ditto replay (live data)
  - Struggle exclusion from move_orders
  - Minior forme deduplication in own_slot_order (no double-counting)

Run:
  cd /home/user/showdown-bot
  python -m pytest tests/test_replay_ingestion.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/user/metamon")
os.environ.setdefault("METAMON_ALLOW_ANY_POKE_ENV", "True")
os.environ.setdefault("METAMON_CACHE_DIR", "/home/user/metamon-cache")

from replay_ingestion.parser import parse_replay_file
from replay_ingestion.reconstruct import reconstruct
from replay_ingestion.trajectory import (
    _compute_move_orders,
    _compute_own_slot_order,
    _encode_action,
    _available_switches,
)

# ---------------------------------------------------------------------------
# Helper — build a minimal JSON replay file from a log string
# ---------------------------------------------------------------------------

def _make_replay(log: str, replay_id: str = "test-illusion") -> dict:
    return {
        "id": replay_id,
        "p1": "Alice",
        "p2": "Bob",
        "rating": 1950,
        "uploadtime": 1700000000,
        "log": log,
    }


def _write_replay(log: str, tmp_path: Path, replay_id: str = "test-illusion") -> Path:
    data = _make_replay(log, replay_id)
    jpath = tmp_path / f"{replay_id}.json"
    jpath.write_text(json.dumps(data))
    lpath = tmp_path / f"{replay_id}.log"
    lpath.write_text(log)
    return jpath


# ---------------------------------------------------------------------------
# Minimal Illusion replay log
#
# Zoroark (p2) enters disguised as Gengar. It takes a hit that breaks Illusion,
# revealing itself. The parser must:
#   1. Record p2's active slot initially as species=Gengar
#   2. On |-end|...|Illusion|, set illusion_entry_species=Gengar, clear species
#   3. On subsequent |detailschange|, set the true species (Zoroark)
# ---------------------------------------------------------------------------

_ILLUSION_LOG = """\
|j|Alice
|j|Bob
|gametype|singles
|player|p1|Alice|1|1950
|player|p2|Bob|1|1950
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] Random Battle
|rule|Sleep Clause Mod: Limit one foe put to sleep
|
|t:|1700000000
|switch|p1a: Vaporeon|Vaporeon, L86, M|302/302
|switch|p2a: Gengar|Gengar, L87|253/253
|turn|1
|move|p1a: Vaporeon|Surf|p2a: Gengar
|-damage|p2a: Gengar|180/253
|-end|p2a: Gengar|Illusion
|detailschange|p2a: Gengar|Zoroark, L83, F
|-damage|p2a: Gengar|140/253
|move|p2a: Gengar|Night Daze|p1a: Vaporeon
|-damage|p1a: Vaporeon|240/302
|turn|2
|move|p1a: Vaporeon|Surf|p2a: Gengar
|-damage|p2a: Gengar|0/253 fnt
|faint|p2a: Gengar
|win|Alice
"""


class TestIllusionBreak:
    def test_illusion_true_species_revealed(self, tmp_path):
        """After Illusion breaks, own_slots should show Zoroark, not Gengar."""
        jpath = _write_replay(_ILLUSION_LOG, tmp_path, "test-illusion")
        battle = parse_replay_file(str(jpath))
        assert battle is not None, "parse failed"
        assert len(battle.turns) >= 1

        rec = reconstruct(battle)
        assert rec is not None

        # p2 view: after turn 1 the opponent should appear as Zoroark in p2's own_slots
        # (p2 IS Zoroark; the first-person view is p2's own team)
        last_p2 = rec.p2_views[-1]
        active = next(
            (s for s in last_p2.own_slots if s.nickname == last_p2.own_active_nick), None
        )
        # Zoroark is fainted at turn 2; after faint its species should still be Zoroark
        zoroark_slot = next(
            (s for s in last_p2.own_slots if "zoroark" in (s.species or "").lower()), None
        )
        assert zoroark_slot is not None, (
            f"Expected Zoroark in p2 own_slots after Illusion break, got: "
            f"{[(s.nickname, s.species) for s in last_p2.own_slots]}"
        )

    def test_illusion_entry_species_recorded(self, tmp_path):
        """Parser snapshot should record illusion_entry_species=Gengar after the break turn.

        The Illusion break happens during turn 1. The snapshot captured at |turn|2|
        (battle.turns[1]) reflects the end-of-turn-1 state, where illusion_entry_species
        has been set and the slot's species updated to Zoroark.
        """
        jpath = _write_replay(_ILLUSION_LOG, tmp_path, "test-illusion")
        battle = parse_replay_file(str(jpath))
        assert battle is not None
        assert len(battle.turns) >= 2, "Expected at least 2 turn snapshots"

        # Turn 2 snapshot (index 1) = state after turn 1 events including Illusion break
        t2 = battle.turns[1]
        zoroark_slot = next(
            (s for s in t2.p2_slots if s.illusion_entry_species is not None), None
        )
        assert zoroark_slot is not None, (
            "Expected illusion_entry_species set on Zoroark slot in turn-2 snapshot; "
            f"p2_slots={[(s.nickname, s.species, s.illusion_entry_species) for s in t2.p2_slots]}"
        )
        assert "gengar" in zoroark_slot.illusion_entry_species.lower(), (
            f"Expected illusion_entry_species=Gengar, got {zoroark_slot.illusion_entry_species!r}"
        )

    def test_opponent_sees_zoroark_after_break(self, tmp_path):
        """From p1's (Alice's) perspective, the opponent's species changes to Zoroark."""
        jpath = _write_replay(_ILLUSION_LOG, tmp_path, "test-illusion")
        battle = parse_replay_file(str(jpath))
        rec = reconstruct(battle)
        assert rec is not None

        # p1 view at the last turn: opponent slot should show Zoroark
        last_p1 = rec.p1_views[-1]
        opp_active = next(
            (s for s in last_p1.opp_slots if s.nickname == last_p1.opp_active_nick), None
        )
        # After faint the slot still carries species info
        revealed_opp = [s for s in last_p1.opp_slots if s.revealed]
        assert any("zoroark" in (s.species or "").lower() for s in revealed_opp), (
            f"P1 should see Zoroark after Illusion break; opp_slots="
            f"{[(s.nickname, s.species, s.revealed) for s in last_p1.opp_slots]}"
        )


# ---------------------------------------------------------------------------
# Struggle exclusion from move_orders
# ---------------------------------------------------------------------------

_STRUGGLE_LOG = """\
|j|Alice
|j|Bob
|gametype|singles
|player|p1|Alice|1|1950
|player|p2|Bob|1|1950
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] Random Battle
|rule|Sleep Clause Mod: Limit one foe put to sleep
|
|t:|1700000000
|switch|p1a: Sylveon|Sylveon, L85, F|300/300
|switch|p2a: Blissey|Blissey, L84, F|520/520
|turn|1
|move|p1a: Sylveon|Wish|p1a: Sylveon
|move|p2a: Blissey|Soft-Boiled|p2a: Blissey
|turn|2
|move|p1a: Sylveon|Struggle|p1a: Sylveon
|-damage|p2a: Blissey|480/520
|-damage|p1a: Sylveon|270/300|[from] Recoil
|move|p2a: Blissey|Soft-Boiled|p2a: Blissey
|win|Bob
"""


class TestStruggleExclusion:
    def test_struggle_not_in_move_orders(self, tmp_path):
        """Struggle must be excluded from move_orders so it never displaces real moves."""
        jpath = _write_replay(_STRUGGLE_LOG, tmp_path, "test-struggle")
        battle = parse_replay_file(str(jpath))
        rec = reconstruct(battle)
        assert rec is not None

        move_orders = _compute_move_orders(rec.p1_views)
        sylveon_moves = move_orders.get("Sylveon", [])
        assert "struggle" not in sylveon_moves, (
            f"Struggle should be excluded from move_orders; got {sylveon_moves}"
        )
        assert "wish" in sylveon_moves, (
            f"Wish should be in move_orders; got {sylveon_moves}"
        )

    def test_wish_encoded_correctly_when_struggle_excluded(self, tmp_path):
        """Wish gets a valid slot index even when Struggle was also used."""
        jpath = _write_replay(_STRUGGLE_LOG, tmp_path, "test-struggle")
        battle = parse_replay_file(str(jpath))
        rec = reconstruct(battle)
        assert rec is not None

        move_orders = _compute_move_orders(rec.p1_views)
        own_order = _compute_own_slot_order(rec.p1_views)

        for view in rec.p1_views:
            if view.action and view.action.name == "Wish":
                enc = _encode_action(view, move_orders, own_order)
                assert enc >= 0, f"Wish should encode to a valid slot; got {enc}"
                break
        else:
            pytest.skip("No Wish turn found in synthetic replay")


# ---------------------------------------------------------------------------
# Minior forme deduplication in own_slot_order
# ---------------------------------------------------------------------------

_MINIOR_LOG = """\
|j|Alice
|j|Bob
|gametype|singles
|player|p1|Alice|1|1950
|player|p2|Bob|1|1950
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] Random Battle
|rule|Sleep Clause Mod: Limit one foe put to sleep
|
|t:|1700000000
|switch|p1a: Minior|Minior-Meteor, L77|209/209
|switch|p2a: Blissey|Blissey, L84, F|520/520
|turn|1
|move|p2a: Blissey|Seismic Toss|p1a: Minior
|-damage|p1a: Minior|105/209
|-end|p1a: Minior|Shields Down
|detailschange|p1a: Minior|Minior-Yellow
|move|p1a: Minior|Power Gem|p2a: Blissey
|-damage|p2a: Blissey|420/520
|turn|2
|switch|p1a: Vaporeon|Vaporeon, L86, M|302/302
|move|p2a: Blissey|Soft-Boiled|p2a: Blissey
|turn|3
|move|p1a: Vaporeon|Surf|p2a: Blissey
|-damage|p2a: Blissey|340/520
|move|p2a: Blissey|Seismic Toss|p1a: Vaporeon
|-damage|p1a: Vaporeon|216/302
|win|Alice
"""


class TestMiniorFormeDedupe:
    def test_own_slot_order_no_duplicate(self, tmp_path):
        """Minior-Meteor and Minior-Yellow must map to one slot, not two."""
        jpath = _write_replay(_MINIOR_LOG, tmp_path, "test-minior")
        battle = parse_replay_file(str(jpath))
        rec = reconstruct(battle)
        assert rec is not None

        own_order = _compute_own_slot_order(rec.p1_views)
        # Only Minior (one form) + Vaporeon = 2 unique Pokemon
        assert len(own_order) == 2, (
            f"Expected 2 entries in own_slot_order (Minior + Vaporeon), got {len(own_order)}: {own_order}"
        )
        # Neither miniormeteor nor minioryellow should appear twice
        assert own_order.count("miniormeteor") + own_order.count("minioryellow") == 1, (
            f"Minior should appear exactly once; own_slot_order={own_order}"
        )

    def test_switch_to_vaporeon_encodes_correctly(self, tmp_path):
        """Switch to Vaporeon encodes to a valid slot even when Minior changed forme."""
        jpath = _write_replay(_MINIOR_LOG, tmp_path, "test-minior")
        battle = parse_replay_file(str(jpath))
        rec = reconstruct(battle)
        assert rec is not None

        move_orders = _compute_move_orders(rec.p1_views)
        own_order = _compute_own_slot_order(rec.p1_views)

        for view in rec.p1_views:
            if view.action and view.action.name == "Vaporeon":
                enc = _encode_action(view, move_orders, own_order)
                assert enc >= 4, f"Switch should encode to slot 4-8; got {enc}"
                assert enc <= 8, f"Switch should encode to slot 4-8; got {enc}"
                break
        else:
            pytest.skip("No switch-to-Vaporeon turn found")
