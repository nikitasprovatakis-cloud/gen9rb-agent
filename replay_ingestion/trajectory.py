"""
D4: Trajectory construction from ReconstructionResult.

For each replay, produces two .npz files (one per player POV) containing:
  states       float32 (T, 959)  — per-turn feature vectors via BattleFeatureExtractor
  actions      int16   (T,)      — action index 0-12 or -1 (unknown/unenforceable)
  legal_masks  bool    (T, 13)   — which action slots are legal this turn
  force_switch bool    (T,)      — True if this is a forced-switch turn
  winner       int8    (1,)      — 1 / 2 / -1 (p1 wins / p2 wins / unknown)
  player       int8    (1,)      — 1 or 2

Action space (13 slots):
  0-3:   move (alphabetical Showdown ID among all moves used this battle)
  4-8:   switch (alphabetical species ID among non-fainted, non-active, revealed own)
  9-12:  tera + move (same ordering as 0-3; legal only if player can still tera)
  -1:    unresolvable (action None, move ordering ambiguous, > 4 moves seen, etc.)

Legality mask uses the FINAL known moveset (all moves seen across entire battle)
for consistent action indexing.  The feature vector still only encodes moves
seen up to the current turn — no future leakage into feature values.

BattleFeatureExtractor compatibility:
  Synthetic duck-typed objects (FakeEnum, SynthMove, SynthPokemon, SynthBattle)
  satisfy every attribute access the extractor makes without importing poke-env
  enum classes directly.  See inline comments for each attribute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from knowledge.set_pool import to_id
from replay_ingestion.reconstruct import ReconstructedView, ReconstructionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy GenData loader (avoids slow import at module load time)
# ---------------------------------------------------------------------------

_GEN_DATA = None
_MOVE_DB: dict = {}


def _get_gen_data():
    global _GEN_DATA, _MOVE_DB
    if _GEN_DATA is None:
        from poke_env.data import GenData
        _GEN_DATA = GenData.from_format("gen9")
        _MOVE_DB = _GEN_DATA.moves
    return _GEN_DATA


# ---------------------------------------------------------------------------
# FakeEnum — duck-typed replacement for poke-env enums
# ---------------------------------------------------------------------------

class FakeEnum:
    """
    Provides the .name interface that BattleFeatureExtractor reads from
    poke-env Weather, Field, SideCondition, Status, PokemonType, MoveCategory
    enum instances.  Hashable so it works as a dict key.
    """
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, FakeEnum):
            return self.name == other.name
        return NotImplemented

    def __repr__(self):
        return f"FakeEnum({self.name!r})"


# ---------------------------------------------------------------------------
# Synthetic poke-env objects
# ---------------------------------------------------------------------------

class SynthMove:
    """Duck-typed poke-env Move (minimal interface for BattleFeatureExtractor)."""
    __slots__ = ("id", "type", "category", "priority", "base_power")

    def __init__(self, move_id: str, type_name: str, category_name: str,
                 priority: int, base_power: int):
        self.id = move_id                           # Showdown ID (already lowercase)
        self.type = FakeEnum(type_name.upper())     # .type.name accessed as e.g. "FIRE"
        self.category = FakeEnum(category_name.upper())   # "PHYSICAL"/"SPECIAL"/"STATUS"
        self.priority = priority
        self.base_power = base_power


class SynthPokemon:
    """
    Duck-typed poke-env Pokemon.

    Attribute notes (matching what BattleFeatureExtractor reads):
      .species              str   — display name or ""
      .current_hp_fraction  float
      .status               FakeEnum|None  — .name == "BRN"/"PAR"/...
      .boosts               dict[str,int]  — keys "atk","def","spa","spd","spe","accuracy","evasion"
      .base_stats           dict[str,int]  — keys "hp","atk","def","spa","spd","spe"
      .moves                dict[str, SynthMove]  — keyed by Showdown move ID
      .ability              str|None
      .item                 str|None
      .is_terastallized     bool
      .tera_type            FakeEnum|None  — .name == e.g. "FIRE"
      .level                int
      .fainted              bool
      .revealed             bool
      .types                list[FakeEnum]  — .name == e.g. "FIRE"
    """

    def __init__(
        self,
        species: str,
        current_hp_fraction: float,
        status_name: Optional[str],
        boosts: dict,
        base_stats: dict,
        moves: dict,
        ability: Optional[str],
        item: Optional[str],
        is_terastallized: bool,
        tera_type_name: Optional[str],
        level: int,
        fainted: bool,
        revealed: bool,
        type_names: list[str],
    ):
        self.species = species
        self.current_hp_fraction = current_hp_fraction
        self.status = FakeEnum(status_name) if status_name else None
        self.boosts = boosts
        self.base_stats = base_stats
        self.moves = moves
        self.ability = ability
        self.item = item
        self.is_terastallized = is_terastallized
        self.tera_type = FakeEnum(tera_type_name.upper()) if tera_type_name else None
        self.level = level
        self.fainted = fainted
        self.revealed = revealed
        self.types = [FakeEnum(t.upper()) for t in type_names]


class SynthBattle:
    """
    Duck-typed poke-env Battle for BattleFeatureExtractor.

    Attribute notes:
      .team                       dict[str, SynthPokemon]  — own team, keyed by species
      .opponent_team              dict[str, SynthPokemon]  — opponent revealed team
      .active_pokemon             SynthPokemon | None
      .opponent_active_pokemon    SynthPokemon | None
      .weather                    set[FakeEnum]  — zero or one element; iterable
      .fields                     dict[FakeEnum, int]  — Field → activation turn
      .side_conditions            dict[FakeEnum, int]  — SideCondition → count/turn
      .opponent_side_conditions   dict[FakeEnum, int]
      .turn                       int
      .battle_tag                 str
    """

    def __init__(
        self,
        team: dict,
        opponent_team: dict,
        active_pokemon: Optional[SynthPokemon],
        opponent_active_pokemon: Optional[SynthPokemon],
        weather: set,
        fields: dict,
        side_conditions: dict,
        opponent_side_conditions: dict,
        turn: int,
    ):
        self.team = team
        self.opponent_team = opponent_team
        self.active_pokemon = active_pokemon
        self.opponent_active_pokemon = opponent_active_pokemon
        self.weather = weather
        self.fields = fields
        self.side_conditions = side_conditions
        self.opponent_side_conditions = opponent_side_conditions
        self.turn = turn
        self.battle_tag = ""


# ---------------------------------------------------------------------------
# Name-mapping constants (parser → poke-env enum .name)
# ---------------------------------------------------------------------------

_WEATHER_ENUM = {
    "RainDance":     "RAINDANCE",
    "SunnyDay":      "SUNNYDAY",
    "Sandstorm":     "SANDSTORM",
    "Snow":          "SNOW",
    "PrimordialSea": "PRIMORDIALSEA",
    "DesolateLand":  "DESOLATELAND",
}

_TERRAIN_ENUM = {
    "Electric": "ELECTRIC_TERRAIN",
    "Grassy":   "GRASSY_TERRAIN",
    "Misty":    "MISTY_TERRAIN",
    "Psychic":  "PSYCHIC_TERRAIN",
}

# parser PokemonSlot.status values → poke-env Status enum .name (uppercase)
_STATUS_ENUM = {
    "brn": "BRN", "psn": "PSN", "tox": "TOX",
    "par": "PAR", "slp": "SLP", "frz": "FRZ",
}

# poke-env move category strings → consistent uppercase
_CATEGORY_ENUM = {
    "Physical": "PHYSICAL", "Special": "SPECIAL", "Status": "STATUS",
    "physical": "PHYSICAL", "special": "SPECIAL", "status": "STATUS",
}


# ---------------------------------------------------------------------------
# GenData lookup helpers
# ---------------------------------------------------------------------------

def _species_base_stats(species: str) -> dict:
    """Look up base stats from poke-env GenData; falls back to neutral defaults."""
    try:
        gd = _get_gen_data()
        sid = to_id(species)
        entry = gd.pokedex.get(sid)
        if entry is None:
            from knowledge.set_pool import resolve_species
            canonical = resolve_species(species)
            if canonical:
                entry = gd.pokedex.get(to_id(canonical))
        if entry and "baseStats" in entry:
            bs = entry["baseStats"]
            return {
                "hp": bs.get("hp", 45),  "atk": bs.get("atk", 45),
                "def": bs.get("def", 45), "spa": bs.get("spa", 45),
                "spd": bs.get("spd", 45), "spe": bs.get("spe", 45),
            }
    except Exception as exc:
        logger.debug("base_stats lookup failed for %r: %s", species, exc)
    return {"hp": 45, "atk": 45, "def": 45, "spa": 45, "spd": 45, "spe": 45}


def _species_types(species: str) -> list[str]:
    """Look up primary type(s) from poke-env GenData; falls back to [NORMAL]."""
    try:
        gd = _get_gen_data()
        sid = to_id(species)
        entry = gd.pokedex.get(sid)
        if entry is None:
            from knowledge.set_pool import resolve_species
            canonical = resolve_species(species)
            if canonical:
                entry = gd.pokedex.get(to_id(canonical))
        if entry and "types" in entry:
            return [t.upper() for t in entry["types"]]
    except Exception as exc:
        logger.debug("types lookup failed for %r: %s", species, exc)
    return ["NORMAL"]


def _build_synth_move(move_name: str) -> SynthMove:
    """Create a SynthMove by looking up the move in poke-env's move database."""
    _get_gen_data()  # ensure _MOVE_DB is populated
    mid = to_id(move_name)
    entry = _MOVE_DB.get(mid, {})
    type_name = (entry.get("type") or "NORMAL").upper()
    cat_raw = entry.get("category") or "Status"
    cat_name = _CATEGORY_ENUM.get(cat_raw, cat_raw.upper())
    priority = int(entry.get("priority") or 0)
    base_power = int(entry.get("basePower") or 0)
    return SynthMove(mid, type_name, cat_name, priority, base_power)


def _build_synth_pokemon(slot) -> SynthPokemon:
    """
    Create a SynthPokemon from a (resolved) PokemonSlot.
    includes only moves that are in slot.moves_used (what has been revealed).
    """
    species = slot.species or ""
    base_stats = _species_base_stats(species) if species else {
        "hp": 45, "atk": 45, "def": 45, "spa": 45, "spd": 45, "spe": 45,
    }
    type_names = _species_types(species) if species else ["NORMAL"]

    moves: dict[str, SynthMove] = {}
    for move_name in slot.moves_used:
        mid = to_id(move_name)
        try:
            moves[mid] = _build_synth_move(move_name)
        except Exception as exc:
            logger.debug("Move build failed for %r: %s", move_name, exc)

    status_name = _STATUS_ENUM.get(slot.status or "") if slot.status else None

    # tera_type_revealed is a capitalised string e.g. "Fire" — convert to uppercase
    tera_type_name: Optional[str] = None
    if slot.is_terastallized and slot.tera_type_revealed:
        tera_type_name = slot.tera_type_revealed.upper()

    return SynthPokemon(
        species=species,
        current_hp_fraction=slot.hp_fraction,
        status_name=status_name,
        boosts=dict(slot.boosts),
        base_stats=base_stats,
        moves=moves,
        ability=slot.ability_revealed,
        item=slot.item_revealed,
        is_terastallized=slot.is_terastallized,
        tera_type_name=tera_type_name,
        level=slot.level,
        fainted=slot.fainted,
        revealed=slot.revealed,
        type_names=type_names,
    )


def _build_side_conditions(side) -> dict:
    """Convert parser SideConditions → {FakeEnum: int} for SynthBattle."""
    conds: dict = {}
    if side.stealth_rock:
        conds[FakeEnum("STEALTH_ROCK")] = 1
    if side.spikes > 0:
        conds[FakeEnum("SPIKES")] = side.spikes
    if side.toxic_spikes > 0:
        conds[FakeEnum("TOXIC_SPIKES")] = side.toxic_spikes
    if side.sticky_web:
        conds[FakeEnum("STICKY_WEB")] = 1
    if side.reflect_turn > 0:
        conds[FakeEnum("REFLECT")] = side.reflect_turn
    if side.light_screen_turn > 0:
        conds[FakeEnum("LIGHT_SCREEN")] = side.light_screen_turn
    if side.aurora_veil_turn > 0:
        conds[FakeEnum("AURORA_VEIL")] = side.aurora_veil_turn
    return conds


def _build_synth_battle(view: ReconstructedView) -> SynthBattle:
    """Build a SynthBattle from one player's first-person view of one turn."""
    own_team: dict[str, SynthPokemon] = {}
    active_mon: Optional[SynthPokemon] = None

    for slot in view.own_slots:
        if not slot.species:
            continue
        mon = _build_synth_pokemon(slot)
        own_team[slot.species] = mon
        if slot.nickname == view.own_active_nick:
            active_mon = mon

    opp_team: dict[str, SynthPokemon] = {}
    opp_active_mon: Optional[SynthPokemon] = None

    for slot in view.opp_slots:
        if not slot.species or not slot.revealed:
            continue
        mon = _build_synth_pokemon(slot)
        opp_team[slot.species] = mon
        if slot.nickname == view.opp_active_nick:
            opp_active_mon = mon

    # Weather
    weather_name = _WEATHER_ENUM.get(view.field_state.weather or "")
    weather: set = {FakeEnum(weather_name)} if weather_name else set()

    # Fields (terrain + trick room)
    fields: dict = {}
    terrain_name = _TERRAIN_ENUM.get(view.field_state.terrain or "")
    if terrain_name:
        fields[FakeEnum(terrain_name)] = view.field_state.terrain_turn
    if view.field_state.trick_room:
        fields[FakeEnum("TRICK_ROOM")] = view.field_state.trick_room_turn

    own_conds = _build_side_conditions(view.own_side)
    opp_conds = _build_side_conditions(view.opp_side)

    return SynthBattle(
        team=own_team,
        opponent_team=opp_team,
        active_pokemon=active_mon,
        opponent_active_pokemon=opp_active_mon,
        weather=weather,
        fields=fields,
        side_conditions=own_conds,
        opponent_side_conditions=opp_conds,
        turn=view.turn_number,
    )


# ---------------------------------------------------------------------------
# Stable own-slot ordering
# ---------------------------------------------------------------------------

def _compute_own_slot_order(views: list[ReconstructedView]) -> list[str]:
    """
    Compute stable own-team slot order (species Showdown IDs, reveal order)
    by scanning all turns of a battle.  Includes all 6 (or fewer) Pokemon
    that appeared during the battle, including those first revealed via a
    switch/drag action on the final turn.

    Tracks by nickname to avoid double-counting Pokemon that change forme
    mid-battle (e.g. Minior-Meteor → Minior-Yellow).  Only the first-seen
    species ID for each nickname is recorded.
    """
    seen_nicks: set[str] = set()
    seen_sids: set[str] = set()
    order: list[str] = []
    for view in views:
        for slot in view.own_slots:
            if slot.species and slot.nickname:
                nick = slot.nickname
                sid = to_id(slot.species)
                if nick not in seen_nicks:
                    seen_nicks.add(nick)
                    if sid not in seen_sids:
                        seen_sids.add(sid)
                        order.append(sid)
        # Also capture species first revealed via action (switch/drag to new Pokemon)
        action = view.action
        if action and action.action_type in ("switch", "drag") and action.name and action.nickname:
            nick = action.nickname
            sid = to_id(action.name)
            if nick not in seen_nicks:
                seen_nicks.add(nick)
                if sid not in seen_sids:
                    seen_sids.add(sid)
                    order.append(sid)
    return order


# ---------------------------------------------------------------------------
# Move ordering and action encoding
# ---------------------------------------------------------------------------

def _compute_move_orders(views: list[ReconstructedView]) -> dict[str, list[str]]:
    """
    Collect all moves used by each own Pokemon (by nickname) across the entire
    battle, including moves from action records.  Returns
    {nickname: [move_id, ...]} sorted alphabetically by Showdown ID.

    This gives consistent 0-3 action indices regardless of when in the battle
    a move was first used.
    """
    # Struggle is automatic (no PP) — exclude it from the action space so it
    # never shifts real moves out of the 0-3 index range.
    _EXCLUDED_MOVES = {"struggle"}

    moves_per_nick: dict[str, set] = {}

    for view in views:
        # From slot.moves_used (accumulated by parser up to this turn's start)
        for slot in view.own_slots:
            if slot.species:
                nick = slot.nickname
                if nick not in moves_per_nick:
                    moves_per_nick[nick] = set()
                for m in slot.moves_used:
                    mid = to_id(m)
                    if mid not in _EXCLUDED_MOVES:
                        moves_per_nick[nick].add(mid)

        # Also capture this turn's action move (not yet in moves_used)
        action = view.action
        if action and action.action_type == "move":
            mid = to_id(action.name)
            if mid not in _EXCLUDED_MOVES:
                nick = view.own_active_nick
                if nick not in moves_per_nick:
                    moves_per_nick[nick] = set()
                moves_per_nick[nick].add(mid)

    return {nick: sorted(moves) for nick, moves in moves_per_nick.items()}


def _available_switches(
    view: ReconstructedView,
    own_slot_order: list[str],
) -> list[str]:
    """
    Return sorted list of species IDs available to switch to this turn.

    Includes:
    - Revealed, non-fainted, non-active own Pokemon (already in snapshot)
    - Unrevealed own Pokemon (first-time switch-in candidates): they appear in
      own_slot_order but not yet in view.own_slots; assumed non-fainted until
      proven otherwise (the player knows their full team from the start).

    Both groups are sorted alphabetically by Showdown species ID so the
    resulting 4-8 action indices are consistent across turns.
    """
    in_snapshot_sids = {to_id(s.species) for s in view.own_slots if s.species}
    active_sid = to_id(
        next((s.species for s in view.own_slots if s.nickname == view.own_active_nick), "")
    )

    candidate_sids: set[str] = set()

    # From snapshot: revealed, non-fainted, non-active
    for s in view.own_slots:
        if s.revealed and not s.fainted and to_id(s.species) != active_sid and s.species:
            candidate_sids.add(to_id(s.species))

    # From own_slot_order: unrevealed (not yet in snapshot) — assumed available
    for sid in own_slot_order:
        if sid not in in_snapshot_sids and sid != active_sid:
            candidate_sids.add(sid)

    return sorted(candidate_sids)


def _encode_action(
    view: ReconstructedView,
    move_orders: dict[str, list[str]],
    own_slot_order: list[str],
) -> int:
    """
    Encode view.action as an int in [-1, 12].
    Returns -1 when the action cannot be encoded (None, ambiguous, >4 moves, etc.).
    """
    action = view.action
    if action is None:
        return -1

    if action.action_type == "move":
        nick = view.own_active_nick
        moves = move_orders.get(nick, [])
        mid = to_id(action.name)
        if mid not in moves:
            return -1
        idx = moves.index(mid)
        if idx > 3:
            return -1
        return (9 + idx) if action.is_tera else idx

    if action.action_type in ("switch", "drag"):
        available = _available_switches(view, own_slot_order)
        target_sid = to_id(action.name)  # action.name = incoming species
        if target_sid not in available:
            return -1
        idx = available.index(target_sid)
        if idx > 4:
            return -1
        return 4 + idx

    return -1


def _compute_legality_mask(
    view: ReconstructedView,
    move_orders: dict[str, list[str]],
    own_slot_order: list[str],
) -> np.ndarray:
    """
    Compute a 13-element bool mask of legal action slots for this turn.
    Uses the final known move order and full team knowledge for switches.
    """
    mask = np.zeros(13, dtype=bool)

    nick = view.own_active_nick
    moves = move_orders.get(nick, [])

    # Move slots 0-3
    for i in range(min(len(moves), 4)):
        mask[i] = True

    # Switch slots 4-8 (using full team knowledge)
    available = _available_switches(view, own_slot_order)
    for i in range(min(len(available), 5)):
        mask[4 + i] = True

    # Tera-move slots 9-12: legal if player can still tera AND move slot is legal
    if view.own_can_tera:
        for i in range(4):
            if mask[i]:
                mask[9 + i] = True

    return mask


# ---------------------------------------------------------------------------
# Main trajectory builder
# ---------------------------------------------------------------------------

class TrajectoryBuilder:
    """
    Converts a ReconstructionResult into per-player .npz trajectory files.

    Must be instantiated once and reused across replays (the extractor is
    reset per battle, not per replay).

    Usage:
        builder = TrajectoryBuilder()
        stats = builder.build_and_save(result, output_dir)
    """

    def __init__(self):
        from knowledge.features import BattleFeatureExtractor
        self._extractor = BattleFeatureExtractor()

    def build_pov(
        self,
        views: list[ReconstructedView],
        winner: Optional[int],
        player: int,
    ) -> dict:
        """
        Build numpy arrays for one player's POV across all turns.

        The extractor is reset before this battle and its own-slot order is
        pre-seeded with the full reveal order so slot indices stay stable.
        """
        from knowledge.features import FEATURE_DIM

        move_orders = _compute_move_orders(views)

        # Pre-set stable own-slot order (all species in reveal order)
        own_order = _compute_own_slot_order(views)
        self._extractor.reset()
        self._extractor._own_slot_order = own_order  # pre-seed before first extract

        n = len(views)
        states = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        actions = np.full(n, -1, dtype=np.int16)
        legal_masks = np.zeros((n, 13), dtype=bool)
        force_switches = np.zeros(n, dtype=bool)

        for t, view in enumerate(views):
            try:
                synth = _build_synth_battle(view)
                states[t] = self._extractor.extract(synth)
            except Exception as exc:
                logger.warning(
                    "Feature extraction failed at turn %d of %s (player %d): %s",
                    view.turn_number, view.replay_id, player, exc,
                )

            try:
                actions[t] = _encode_action(view, move_orders, own_order)
                legal_masks[t] = _compute_legality_mask(view, move_orders, own_order)
            except Exception as exc:
                logger.warning(
                    "Action encoding failed at turn %d: %s", view.turn_number, exc
                )

            force_switches[t] = view.is_force_switch

        winner_val = winner if winner in (1, 2) else -1
        return {
            "states": states,
            "actions": actions,
            "legal_masks": legal_masks,
            "force_switch": force_switches,
            "winner": np.array([winner_val], dtype=np.int8),
            "player": np.array([player], dtype=np.int8),
        }

    def build_and_save(
        self,
        result: ReconstructionResult,
        output_dir: str | Path,
    ) -> dict:
        """
        Build trajectories for both POVs and save as compressed .npz files.

        Files created:
          {output_dir}/{replay_id}_p1.npz
          {output_dir}/{replay_id}_p2.npz

        Returns a stats dict.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rid = result.replay_id

        saved = 0
        errors = []

        for player, views in ((1, result.p1_views), (2, result.p2_views)):
            try:
                data = self.build_pov(views, result.winner, player)
                out_path = output_dir / f"{rid}_p{player}.npz"
                np.savez_compressed(out_path, **data)
                saved += 1
                logger.debug("Saved %s", out_path)
            except Exception as exc:
                logger.error("Trajectory build failed for %s p%d: %s", rid, player, exc)
                errors.append(f"p{player}: {exc}")

        return {
            "replay_id": rid,
            "turns": len(result.p1_views),
            "npz_files_saved": saved,
            "winner": result.winner,
            "errors": errors,
        }
