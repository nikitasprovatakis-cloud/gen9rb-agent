"""
D3: Battle feature extractor — Option C encoding.

Converts a poke-env Battle object into a fixed-size numpy feature vector.
Expected values are computed over the SetPredictor distribution for opponent
Pokemon. Own Pokemon are known exactly (probabilities collapse to 0/1).

Feature vector layout (FEATURE_DIM = 947):
  Per Pokemon × 12 (6 own + 6 opponent):     76 each = 912
  Global field features:                              = 35
  Total:                                            = 947

Per-Pokemon features (76):
  [0]       species_idx          — index into ALL_SPECIES list (0 = unknown)
  [1]       hp_fraction          — current HP %
  [2:9]     status_onehot        — 7 values: none/brn/psn/tox/par/slp/frz
  [9:16]    stat_boosts          — 7 values: atk/def/spa/spd/spe/acc/eva (-6 to +6)
  [16:22]   expected_base_stats  — 6 base stats normalized by /255
  [22]      expected_speed_tier  — raw speed stat / 200 (normalized)
  [23:41]   type_move_probs      — 18 type probabilities
  [41]      priority_prob
  [42]      setup_prob
  [43]      hazard_prob
  [44]      removal_prob
  [45]      pivot_prob
  [46:54]   item_probs           — 8 key items
  [54]      expected_damage_norm — expected damage / 200 (normalized)
  [55]      tera_available       — binary
  [56:74]   tera_type_dist       — 18 type probabilities
  [74]      times_active         — integer (capped at 10, then /10)
  [75]      is_revealed          — 1.0 if opponent Pokemon has been seen

Global features (35):
  [0:5]     weather              — 5 one-hot: none/sun/rain/sand/snow
  [5:10]    terrain              — 5 one-hot: none/electric/grassy/misty/psychic
  [10:12]   trick_room           — [active binary, turns_remaining/5]
  [12:16]   our_hazards          — SR, spikes/3, tspikes/2, web
  [16:20]   opp_hazards
  [20:26]   our_screens          — reflect [active, turns/8], ls [active, turns/8], veil [active, turns/8]
  [26:32]   opp_screens
  [32]      turn_number/100      — normalized
  [33]      our_remaining/6
  [34]      opp_remaining/6
"""

import logging
import time
from typing import Optional

import numpy as np

from knowledge.set_pool import get_all_species, get_species_data, resolve_species, to_id
from knowledge.set_predictor import SetPredictor
from knowledge.formes import FormeManager, is_transitional
from knowledge.damage_calc import DamageCalculator, PokemonState, MoveState, FieldState

logger = logging.getLogger(__name__)

# ── Move classification sets ────────────────────────────────────────────────
# Showdown IDs (lowercase, no non-alphanumeric)

SETUP_MOVES = frozenset({
    "swordsdance", "calmmind", "nastyplot", "dragondance", "bulkup", "quiverdance",
    "shellsmash", "shiftgear", "tailglow", "geomancy", "coilingcurse",
    "tidyup", "filletaway", "victory dance", "victorydance",
    "agility", "autotomize", "rockpolish", "speedboost",
    "irondefense", "amnesia", "batonpass",
})

HAZARD_MOVES = frozenset({
    "stealthrock", "spikes", "toxicspikes", "stickyweb",
})

HAZARD_REMOVAL_MOVES = frozenset({
    "rapidspin", "defog", "courtchange", "tidyup", "mortalspinstrike",
})

PIVOT_MOVES = frozenset({
    "uturn", "voltswitch", "flipturn", "partingshot", "teleport", "batonpass",
})

PRIORITY_THRESHOLD = 1  # move.priority >= this is "priority"

KEY_ITEMS = [
    "choiceband",
    "choicespecs",
    "choicescarf",
    "lifeorb",
    "leftovers",
    "assaultvest",
    "heavydutyboots",
    "focussash",
]

TYPES_ORDERED = [
    "BUG", "DARK", "DRAGON", "ELECTRIC", "FAIRY", "FIGHTING",
    "FIRE", "FLYING", "GHOST", "GRASS", "GROUND", "ICE",
    "NORMAL", "POISON", "PSYCHIC", "ROCK", "STEEL", "WATER",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES_ORDERED)}

STATUS_ORDER = ["none", "BRN", "PSN", "TOX", "PAR", "SLP", "FRZ"]
STATUS_TO_IDX = {s: i for i, s in enumerate(STATUS_ORDER)}

WEATHER_ORDER = ["none", "sun", "rain", "sand", "snow"]
TERRAIN_ORDER = ["none", "electric", "grassy", "misty", "psychic"]

# poke-env enum name → canonical label
WEATHER_MAP = {
    "SUNNYDAY": "sun", "DESOLATELAND": "sun",
    "RAINDANCE": "rain", "PRIMORDIALSEA": "rain",
    "SANDSTORM": "sand",
    "SNOW": "snow", "SNOWSCAPE": "snow", "HAIL": "snow",
}
TERRAIN_MAP = {
    "ELECTRIC_TERRAIN": "electric",
    "GRASSY_TERRAIN": "grassy",
    "MISTY_TERRAIN": "misty",
    "PSYCHIC_TERRAIN": "psychic",
}

POKEMON_FEATURES = 76
GLOBAL_FEATURES = 35
TEAM_SIZE = 6
FEATURE_DIM = POKEMON_FEATURES * TEAM_SIZE * 2 + GLOBAL_FEATURES  # = 947


def _build_species_list() -> list[str]:
    return ["__unknown__"] + get_all_species()


class BattleFeatureExtractor:
    """
    Converts a poke-env Battle into a fixed-size float32 numpy vector of length FEATURE_DIM.

    Must be instantiated once and reused across turns of the same battle.
    Call reset() at the start of each new battle.
    """

    FEATURE_DIM = FEATURE_DIM

    def __init__(self):
        self._species_list = _build_species_list()
        self._species_to_idx = {s: i for i, s in enumerate(self._species_list)}
        self._calc = DamageCalculator()
        self._move_db: dict = {}
        self._load_move_db()

        # Per-battle state (reset each battle)
        self._opp_predictors: dict[str, SetPredictor] = {}  # species_id → predictor
        self._times_active: dict[str, int] = {}             # "own:species" / "opp:species" → count
        self._last_active: dict[str, str] = {}              # "own"/"opp" → species_id of last active
        self._forme_manager = FormeManager()
        self._prev_moves_seen: dict[str, set] = {}          # opp species_id → set of observed moves
        self._prev_items_seen: dict[str, bool] = {}
        self._prev_abilities_seen: dict[str, bool] = {}
        self._feature_dim_verified = False

    def _load_move_db(self) -> None:
        try:
            from poke_env.data import GenData
            gd = GenData.from_format("gen9")
            self._move_db = gd.moves
        except Exception as e:
            logger.warning("Could not load poke-env move DB: %s", e)
            self._move_db = {}

    def reset(self) -> None:
        """Call at the start of each new battle."""
        self._opp_predictors.clear()
        self._times_active.clear()
        self._last_active.clear()
        self._forme_manager.reset()
        self._prev_moves_seen.clear()
        self._prev_items_seen.clear()
        self._prev_abilities_seen.clear()

    # ── set predictor management ───────────────────────────────────────────

    def _get_predictor(self, species: str) -> Optional[SetPredictor]:
        canonical = resolve_species(species)
        if canonical is None:
            return None
        sid = to_id(canonical)
        if sid not in self._opp_predictors:
            try:
                self._opp_predictors[sid] = SetPredictor(canonical)
                self._prev_moves_seen[sid] = set()
                self._prev_items_seen[sid] = False
                self._prev_abilities_seen[sid] = False
            except Exception as e:
                logger.warning("Could not create SetPredictor for %s: %s", species, e)
                return None
        return self._opp_predictors[sid]

    def _update_predictor_from_pokemon(self, pokemon) -> None:
        """Feed newly observed info into the SetPredictor for an opponent Pokemon."""
        sid = to_id(pokemon.species)
        pred = self._get_predictor(pokemon.species)
        if pred is None:
            return
        seen_moves = self._prev_moves_seen.setdefault(sid, set())
        for move in pokemon.moves.values():
            mid = to_id(move.id if hasattr(move, "id") else str(move))
            if mid not in seen_moves:
                seen_moves.add(mid)
                try:
                    pred.observe_move(mid)
                except Exception:
                    pass
        if pokemon.item and not self._prev_items_seen.get(sid):
            self._prev_items_seen[sid] = True
            try:
                pred.observe_item(pokemon.item)
            except Exception:
                pass
        if pokemon.ability and not self._prev_abilities_seen.get(sid):
            self._prev_abilities_seen[sid] = True
            try:
                pred.observe_ability(pokemon.ability)
            except Exception:
                pass

    # ── species index ──────────────────────────────────────────────────────

    def _species_idx(self, species: str) -> int:
        canonical = resolve_species(species)
        if canonical is None:
            return 0
        return self._species_to_idx.get(canonical, 0)

    # ── per-Pokemon encoding ───────────────────────────────────────────────

    def _encode_own_pokemon(self, mon, battle) -> np.ndarray:
        """Encode own Pokemon where all info is known exactly."""
        v = np.zeros(POKEMON_FEATURES, dtype=np.float32)

        # Species
        v[0] = self._species_idx(mon.species)

        # HP
        v[1] = float(mon.current_hp_fraction)

        # Status
        status_name = mon.status.name if mon.status else "none"
        v[2 + STATUS_TO_IDX.get(status_name, 0)] = 1.0

        # Stat boosts
        for i, stat in enumerate(("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")):
            v[9 + i] = float(mon.boosts.get(stat, 0)) / 6.0

        # Base stats (normalized by /255)
        for i, stat in enumerate(("hp", "atk", "def", "spa", "spd", "spe")):
            bs = mon.base_stats.get(stat, 0)
            # Apply forme override if applicable
            fs = self._forme_manager.effective_base_stats("own", mon.species,
                                                           mon.base_stats)
            v[16 + i] = float(fs.get(stat, bs)) / 255.0

        # Speed tier (raw stat / 200)
        spe_base = self._forme_manager.effective_base_stats("own", mon.species,
                                                              mon.base_stats).get("spe",
                                                              mon.base_stats.get("spe", 0))
        v[22] = float(spe_base) / 200.0

        # Move type probabilities (exact, 0/1 per type)
        for move in mon.moves.values():
            tname = move.type.name if hasattr(move.type, "name") else str(move.type)
            if tname in TYPE_TO_IDX:
                v[23 + TYPE_TO_IDX[tname]] = 1.0

        # Priority move probability (exact)
        for move in mon.moves.values():
            if getattr(move, "priority", 0) >= PRIORITY_THRESHOLD:
                v[41] = 1.0
                break

        # Functional move probabilities
        for move in mon.moves.values():
            mid = to_id(move.id if hasattr(move, "id") else str(move))
            if mid in SETUP_MOVES:
                v[42] = 1.0
            if mid in HAZARD_MOVES:
                v[43] = 1.0
            if mid in HAZARD_REMOVAL_MOVES:
                v[44] = 1.0
            if mid in PIVOT_MOVES:
                v[45] = 1.0

        # Item probabilities (8 key items, 0/1)
        item_id = to_id(mon.item or "")
        for i, key_item in enumerate(KEY_ITEMS):
            v[46 + i] = 1.0 if item_id == key_item else 0.0

        # Expected damage (own → generic opponent, best move / 200)
        v[54] = self._best_move_damage_norm(mon, battle)

        # Tera
        v[55] = 1.0 if not mon.is_terastallized else 0.0
        if mon.tera_type is not None:
            tname = mon.tera_type.name if hasattr(mon.tera_type, "name") else str(mon.tera_type)
            if tname in TYPE_TO_IDX:
                v[56 + TYPE_TO_IDX[tname]] = 1.0
        elif mon.is_terastallized:
            # Already used, type revealed via is_terastallized + types
            pass

        # Times active
        key = f"own:{to_id(mon.species)}"
        v[74] = min(self._times_active.get(key, 0), 10) / 10.0
        v[75] = 1.0  # own Pokemon always revealed

        return v

    def _encode_opp_pokemon(self, mon, is_active: bool) -> np.ndarray:
        """Encode opponent Pokemon using SetPredictor distribution for unknowns."""
        v = np.zeros(POKEMON_FEATURES, dtype=np.float32)
        revealed = mon.revealed if hasattr(mon, "revealed") else (mon.species != "")
        v[75] = 1.0 if revealed else 0.0

        if not revealed:
            return v

        # Update predictor from newly seen info
        self._update_predictor_from_pokemon(mon)

        pred = self._get_predictor(mon.species)

        # Species
        v[0] = self._species_idx(mon.species)

        # HP
        v[1] = float(mon.current_hp_fraction)

        # Status
        status_name = mon.status.name if mon.status else "none"
        v[2 + STATUS_TO_IDX.get(status_name, 0)] = 1.0

        # Stat boosts
        for i, stat in enumerate(("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")):
            v[9 + i] = float(mon.boosts.get(stat, 0)) / 6.0

        # Expected base stats from randbats
        if pred is not None:
            try:
                species_data = get_species_data(mon.species)
                bs = mon.base_stats  # poke-env always has these
                # Forme override
                fs = self._forme_manager.effective_base_stats("opp", mon.species, bs)
                for i, stat in enumerate(("hp", "atk", "def", "spa", "spd", "spe")):
                    v[16 + i] = float(fs.get(stat, bs.get(stat, 0))) / 255.0
            except Exception:
                for i, stat in enumerate(("hp", "atk", "def", "spa", "spd", "spe")):
                    v[16 + i] = float(mon.base_stats.get(stat, 0)) / 255.0
        else:
            for i, stat in enumerate(("hp", "atk", "def", "spa", "spd", "spe")):
                v[16 + i] = float(mon.base_stats.get(stat, 0)) / 255.0

        # Speed tier
        spe = mon.base_stats.get("spe", 0)
        if pred is not None:
            try:
                fs = self._forme_manager.effective_base_stats("opp", mon.species, mon.base_stats)
                spe = fs.get("spe", spe)
            except Exception:
                pass
            # Scarf boost: weighted by P(has choice scarf)
            item_exp = pred.expected_attr("items")
            scarf_prob = sum(v for k, v in item_exp.items() if to_id(k) == "choicescarf")
            spe_eff = spe * (1.0 + 0.5 * scarf_prob)
        else:
            spe_eff = spe
        v[22] = min(spe_eff / 200.0, 2.0)

        # Move type probabilities
        if pred is not None:
            type_probs = pred.expected_move_type_probs(self._move_db)
            for type_name, prob in type_probs.items():
                tname_up = type_name.upper()
                if tname_up in TYPE_TO_IDX:
                    v[23 + TYPE_TO_IDX[tname_up]] = float(prob)

        # Priority / setup / hazard / removal / pivot probabilities
        if pred is not None:
            priority_ids = {to_id(k) for k, mv in self._move_db.items()
                            if mv.get("priority", 0) >= PRIORITY_THRESHOLD}
            v[41] = pred.prob_has_flag(priority_ids, self._move_db)
            v[42] = pred.prob_has_flag(SETUP_MOVES, self._move_db)
            v[43] = pred.prob_has_flag(HAZARD_MOVES, self._move_db)
            v[44] = pred.prob_has_flag(HAZARD_REMOVAL_MOVES, self._move_db)
            v[45] = pred.prob_has_flag(PIVOT_MOVES, self._move_db)

        # Key item probabilities
        if pred is not None:
            item_exp = pred.expected_attr("items")
            for i, key_item in enumerate(KEY_ITEMS):
                v[46 + i] = float(
                    sum(fq for nm, fq in item_exp.items() if to_id(nm) == key_item)
                )

        # Expected damage
        v[54] = 0.5  # placeholder: opponent's expected damage unknown without full calc

        # Tera availability and type distribution
        v[55] = 0.0 if mon.is_terastallized else 1.0
        if mon.is_terastallized and mon.tera_type is not None:
            tname = mon.tera_type.name if hasattr(mon.tera_type, "name") else str(mon.tera_type)
            if tname in TYPE_TO_IDX:
                v[56 + TYPE_TO_IDX[tname]] = 1.0
        elif pred is not None:
            tera_exp = pred.expected_attr("teraTypes")
            for tname, prob in tera_exp.items():
                tup = tname.upper()
                if tup in TYPE_TO_IDX:
                    v[56 + TYPE_TO_IDX[tup]] += float(prob)

        # Times active
        key = f"opp:{to_id(mon.species)}"
        v[74] = min(self._times_active.get(key, 0), 10) / 10.0

        return v

    def _best_move_damage_norm(self, mon, battle) -> float:
        """Estimate best move expected damage against a generic mid-tier opponent (normalized /200)."""
        if not mon.moves:
            return 0.0
        opp = battle.opponent_active_pokemon
        if opp is None:
            return 0.0
        best = 0.0
        for move in mon.moves.values():
            if getattr(move, "category", None) and move.category.name == "STATUS":
                continue
            bp = getattr(move, "base_power", 0)
            if bp == 0:
                continue
            tname = move.type.name if hasattr(move.type, "name") else "NORMAL"
            stab = 1.5 if tname in [t.name for t in mon.types] else 1.0
            eff = opp.damage_multiplier(move) if opp else 1.0
            score = bp * stab * eff
            best = max(best, score)
        return min(best / 200.0, 2.0)

    # ── global field encoding ──────────────────────────────────────────────

    def _encode_field(self, battle) -> np.ndarray:
        v = np.zeros(GLOBAL_FEATURES, dtype=np.float32)

        # Weather
        weather_label = "none"
        for w in battle.weather:
            weather_label = WEATHER_MAP.get(w.name, "none")
            break
        if weather_label in WEATHER_ORDER:
            v[WEATHER_ORDER.index(weather_label)] = 1.0
        else:
            v[0] = 1.0  # none

        # Terrain
        terrain_label = "none"
        for f in battle.fields:
            label = TERRAIN_MAP.get(f.name, "")
            if label:
                terrain_label = label
                break
        if terrain_label in TERRAIN_ORDER:
            v[5 + TERRAIN_ORDER.index(terrain_label)] = 1.0
        else:
            v[5] = 1.0  # none

        # Trick Room
        from poke_env.environment import Field
        trick_active = any(f.name == "TRICK_ROOM" for f in battle.fields)
        v[10] = 1.0 if trick_active else 0.0
        if trick_active:
            activation_turn = next((t for f, t in battle.fields.items()
                                    if f.name == "TRICK_ROOM"), 0)
            remaining = max(0, 5 - (battle.turn - activation_turn))
            v[11] = remaining / 5.0

        # Hazards
        from poke_env.environment import SideCondition
        def hazard_vec(side_conds):
            hv = np.zeros(4, dtype=np.float32)
            hv[0] = 1.0 if any(s.name == "STEALTH_ROCK" for s in side_conds) else 0.0
            spikes = next((v for s, v in side_conds.items() if s.name == "SPIKES"), 0)
            hv[1] = spikes / 3.0
            tspikes = next((v for s, v in side_conds.items() if s.name == "TOXIC_SPIKES"), 0)
            hv[2] = tspikes / 2.0
            hv[3] = 1.0 if any(s.name == "STICKY_WEB" for s in side_conds) else 0.0
            return hv

        v[12:16] = hazard_vec(battle.side_conditions)
        v[16:20] = hazard_vec(battle.opponent_side_conditions)

        # Screens
        def screen_vec(side_conds, turn):
            sv = np.zeros(6, dtype=np.float32)
            for name, offset, duration in (
                ("REFLECT", 0, 5),
                ("LIGHT_SCREEN", 2, 5),
                ("AURORA_VEIL", 4, 8),
            ):
                active = any(s.name == name for s in side_conds)
                sv[offset] = 1.0 if active else 0.0
                if active:
                    act_turn = next((t for s, t in side_conds.items() if s.name == name), turn)
                    rem = max(0, duration - (turn - act_turn))
                    sv[offset + 1] = rem / duration
            return sv

        v[20:26] = screen_vec(battle.side_conditions, battle.turn)
        v[26:32] = screen_vec(battle.opponent_side_conditions, battle.turn)

        # Turn number
        v[32] = min(battle.turn / 100.0, 1.0)

        # Pokemon remaining
        our_rem = sum(1 for m in battle.team.values() if not m.fainted)
        opp_rem = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
        v[33] = our_rem / 6.0
        v[34] = opp_rem / 6.0

        return v

    # ── update activity counters ───────────────────────────────────────────

    def _update_activity(self, battle) -> None:
        own_active = battle.active_pokemon
        if own_active:
            key = f"own:{to_id(own_active.species)}"
            prev = self._last_active.get("own")
            if prev != to_id(own_active.species):
                self._last_active["own"] = to_id(own_active.species)
                self._times_active[key] = self._times_active.get(key, 0) + 1
                if is_transitional(own_active.species):
                    self._forme_manager.get("own", own_active.species).on_switch_in()
        opp_active = battle.opponent_active_pokemon
        if opp_active:
            key = f"opp:{to_id(opp_active.species)}"
            prev = self._last_active.get("opp")
            if prev != to_id(opp_active.species):
                if prev:
                    self._forme_manager.get("opp", prev).on_switch_out()
                self._last_active["opp"] = to_id(opp_active.species)
                self._times_active[key] = self._times_active.get(key, 0) + 1
                if is_transitional(opp_active.species):
                    self._forme_manager.get("opp", opp_active.species).on_switch_in()
            # HP-based forme transitions
            if is_transitional(opp_active.species):
                self._forme_manager.get("opp", opp_active.species).on_damage_taken(
                    opp_active.current_hp_fraction
                )

    # ── main extraction ────────────────────────────────────────────────────

    def extract(self, battle) -> np.ndarray:
        """
        Extract the full feature vector from a poke-env Battle object.
        Returns float32 array of shape (FEATURE_DIM,).
        """
        t0 = time.perf_counter()
        self._update_activity(battle)

        vec = np.zeros(FEATURE_DIM, dtype=np.float32)

        # ── Own team (slots 0-5) ──────────────────────────────────────────
        own_active = battle.active_pokemon
        own_mons = sorted(battle.team.values(),
                          key=lambda m: (0 if m.active else 1, to_id(m.species)))
        for slot, mon in enumerate(own_mons[:TEAM_SIZE]):
            start = slot * POKEMON_FEATURES
            if not mon.fainted:
                vec[start:start + POKEMON_FEATURES] = self._encode_own_pokemon(mon, battle)

        # ── Opponent team (slots 6-11) ────────────────────────────────────
        opp_active = battle.opponent_active_pokemon
        opp_mons = sorted(battle.opponent_team.values(),
                          key=lambda m: (0 if m.active else 1, to_id(m.species)))
        for slot, mon in enumerate(opp_mons[:TEAM_SIZE]):
            start = (TEAM_SIZE + slot) * POKEMON_FEATURES
            vec[start:start + POKEMON_FEATURES] = self._encode_opp_pokemon(mon, mon.active)

        # ── Global field ──────────────────────────────────────────────────
        vec[TEAM_SIZE * 2 * POKEMON_FEATURES:] = self._encode_field(battle)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 50.0:
            logger.warning("Feature extraction took %.1f ms (> 50 ms target)", elapsed_ms)

        # Validate on first call
        if not self._feature_dim_verified:
            assert vec.shape == (FEATURE_DIM,), f"Shape mismatch: {vec.shape}"
            assert not np.any(np.isnan(vec)), "NaN in feature vector"
            assert not np.any(np.isinf(vec)), "Inf in feature vector"
            self._feature_dim_verified = True

        return vec
