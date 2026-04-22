"""
D3: Battle feature extractor — Option C encoding.

Converts a poke-env Battle object into a fixed-size numpy feature vector.
Expected values are computed over the SetPredictor distribution for opponent
Pokemon. Own Pokemon are known exactly (probabilities collapse to 0/1).

Feature vector layout (FEATURE_DIM = 959):
  Per Pokemon × 12 (6 own + 6 opponent):     77 each = 924
  Global field features:                              = 35
  Total:                                            = 959

SLOT ORDERING — stable identity across turns:
  Own  slots 0-5:   initial team order as given by poke-env at battle start.
                    Slot 0 = team member 1, ..., slot 5 = team member 6.
                    Never changes within a battle.
  Opp  slots 6-11:  reveal order. Slot 6 = first opponent seen, slot 7 = second, etc.
                    Unrevealed slots remain zero-filled (is_revealed=0, is_active=0).
  is_active at index [2] indicates the currently active Pokemon in each slot.
  Active status is no longer the sort key.

Per-Pokemon features (77):
  [0]       species_idx          — index into ALL_SPECIES list (0 = unknown/unrevealed)
  [1]       hp_fraction          — current HP %
  [2]       is_active            — 1.0 if this is the currently active Pokemon
  [3:10]    status_onehot        — 7 values: none/brn/psn/tox/par/slp/frz
  [10:17]   stat_boosts          — 7 values: atk/def/spa/spd/spe/acc/eva (-6 to +6, /6)
  [17:23]   expected_base_stats  — 6 base stats normalized by /255
  [23]      expected_speed_tier  — raw speed stat / 200 (scarf-weighted for opp)
  [24:42]   type_move_probs      — 18 type probabilities
  [42]      priority_prob
  [43]      setup_prob
  [44]      hazard_prob
  [45]      removal_prob
  [46]      pivot_prob
  [47:55]   item_probs           — 8 key items
  [55]      expected_damage_norm — expected damage as fraction of opponent max HP
  [56]      tera_available       — binary (1 if not yet terastallized)
  [57:75]   tera_type_dist       — 18 type probabilities
  [75]      times_active         — switch-in count (capped at 10, then /10)
  [76]      is_revealed          — 1.0 if opponent Pokemon has been seen this battle

Global features (35):
  [0:5]     weather              — 5 one-hot: none/sun/rain/sand/snow
  [5:10]    terrain              — 5 one-hot: none/electric/grassy/misty/psychic
  [10:12]   trick_room           — [active binary, turns_remaining/5]
  [12:16]   our_hazards          — SR, spikes/3, tspikes/2, web
  [16:20]   opp_hazards
  [20:26]   our_screens          — reflect [active, rem/5], ls [active, rem/5], veil [active, rem/8]
  [26:32]   opp_screens
  [32]      turn_number/100      — normalized
  [33]      our_remaining/6
  [34]      opp_remaining/6      — estimated from opponent_team (fainted always revealed)

SPECIES ENCODING NOTE:
  species_idx is a single integer in [0, 508], not a one-hot vector. This is
  correct for use with nn.Embedding in Phase 4's policy network. Do NOT feed
  this index as a continuous float into a linear layer — it is meaningless as
  a continuous value. Phase 4 network design must handle slot [0] of each
  Pokemon block with an Embedding lookup before concatenating with the rest
  of the 76 continuous features.

FEATURE [55] NOTE:
  expected_damage_norm uses the full DamageCalculator (not an inline formula).
  It estimates own active Pokemon's best move as a fraction of the opponent's
  inferred max HP. The DamageCalculator handles STAB, type effectiveness, items,
  abilities, and weather. Intentionally NOT unified with the inline heuristic
  that was used in earlier drafts — do not simplify back to bp*stab*eff/200.
"""

import logging
import time
from typing import Optional

import numpy as np

from knowledge.set_pool import get_all_species, get_species_data, resolve_species, to_id
from knowledge.set_predictor import SetPredictor
from knowledge.formes import FormeManager, is_transitional
from knowledge.damage_calc import (
    DamageCalculator, PokemonState, MoveState, FieldState, _infer_stat, _effective_stat
)

logger = logging.getLogger(__name__)

# ── Move classification sets ────────────────────────────────────────────────
# All Showdown IDs (lowercase, no non-alphanumeric)

SETUP_MOVES = frozenset({
    "swordsdance", "calmmind", "nastyplot", "dragondance", "bulkup", "quiverdance",
    "shellsmash", "shiftgear", "tailglow", "geomancy",
    "tidyup", "filletaway", "victorydance",
    "agility", "autotomize", "rockpolish",
    "irondefense", "amnesia", "batonpass",
    "coil",          # Arbok/Serperior: raises Atk/Def/Acc
})

HAZARD_MOVES = frozenset({
    "stealthrock", "spikes", "toxicspikes", "stickyweb",
})

HAZARD_REMOVAL_MOVES = frozenset({
    "rapidspin", "defog", "courtchange", "tidyup", "mortalspin",
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

# poke-env Weather enum name → feature label
WEATHER_MAP = {
    "SUNNYDAY": "sun", "DESOLATELAND": "sun",
    "RAINDANCE": "rain", "PRIMORDIALSEA": "rain",
    "SANDSTORM": "sand",
    "SNOW": "snow", "SNOWSCAPE": "snow", "HAIL": "snow",
    "DELTASTREAM": "none",   # Rayquaza's weather; no standard damage modifier
}

# poke-env Field enum name → feature label
TERRAIN_MAP = {
    "ELECTRIC_TERRAIN": "electric",
    "GRASSY_TERRAIN": "grassy",
    "MISTY_TERRAIN": "misty",
    "PSYCHIC_TERRAIN": "psychic",
}

# poke-env weather name → damage_calc.py FieldState weather string
_WEATHER_TO_FIELDSTATE = {
    "RAINDANCE": "RainDance",
    "PRIMORDIALSEA": "PrimordialSea",
    "SUNNYDAY": "SunnyDay",
    "DESOLATELAND": "DesolateLand",
    "SANDSTORM": "Sandstorm",
    "SNOW": "Snow",
    "SNOWSCAPE": "Snowscape",
    "HAIL": "Snow",
}

POKEMON_FEATURES = 77   # one more than before (is_active added at [2])
GLOBAL_FEATURES = 35
TEAM_SIZE = 6
FEATURE_DIM = POKEMON_FEATURES * TEAM_SIZE * 2 + GLOBAL_FEATURES  # = 959


def _build_species_list() -> list[str]:
    return ["__unknown__"] + get_all_species()


class BattleFeatureExtractor:
    """
    Converts a poke-env Battle into a fixed-size float32 numpy vector of length FEATURE_DIM.

    Must be instantiated once per session and reused across turns of the same battle.
    Call reset() at the start of each new battle.

    Slot ordering guarantee:
      - Own slots 0-5 reflect the initial team order from battle start (stable).
      - Opp slots 6-11 reflect reveal order (first seen = slot 6, etc., stable).
    """

    FEATURE_DIM = FEATURE_DIM

    def __init__(self):
        self._species_list = _build_species_list()
        self._species_to_idx = {s: i for i, s in enumerate(self._species_list)}
        self._calc = DamageCalculator()
        self._move_db: dict = {}
        self._load_move_db()

        # Per-battle state (reset each battle)
        self._opp_predictors: dict[str, SetPredictor] = {}
        self._times_active: dict[str, int] = {}
        self._last_active: dict[str, str] = {}
        self._forme_manager = FormeManager()
        self._prev_moves_seen: dict[str, set] = {}
        self._prev_items_seen: dict[str, bool] = {}
        self._prev_abilities_seen: dict[str, bool] = {}
        # Stable slot ordering
        self._own_slot_order: Optional[list[str]] = None   # species_ids, set once at battle start
        self._opp_slot_order: list[str] = []               # species_ids in reveal order
        self._opp_slot_set: set[str] = set()               # fast membership check
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
        self._own_slot_order = None
        self._opp_slot_order = []
        self._opp_slot_set = set()

    # ── slot management ────────────────────────────────────────────────────

    def _init_own_slots(self, battle) -> None:
        """Capture own team order from poke-env's insertion order at battle start.
        Called exactly once per battle (on first extract). Never mutated after."""
        self._own_slot_order = [to_id(m.species) for m in battle.team.values()]

    def _update_opp_slots(self, battle) -> None:
        """Extend opponent reveal order as new Pokemon appear. Order is stable."""
        for mon in battle.opponent_team.values():
            sid = to_id(mon.species)
            if sid not in self._opp_slot_set:
                self._opp_slot_order.append(sid)
                self._opp_slot_set.add(sid)

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

    # ── field state builder ────────────────────────────────────────────────

    def _build_field_state(self, battle) -> FieldState:
        """Extract a FieldState from the live battle for DamageCalculator."""
        weather = None
        for w in battle.weather:
            weather = _WEATHER_TO_FIELDSTATE.get(w.name)
            break

        opp_conds = battle.opponent_side_conditions
        own_conds = battle.side_conditions

        return FieldState(
            weather=weather,
            defender_side_reflect=any(s.name == "REFLECT" for s in opp_conds),
            defender_side_light_screen=any(s.name == "LIGHT_SCREEN" for s in opp_conds),
            defender_side_aurora_veil=any(s.name == "AURORA_VEIL" for s in opp_conds),
            attacker_side_reflect=any(s.name == "REFLECT" for s in own_conds),
            attacker_side_light_screen=any(s.name == "LIGHT_SCREEN" for s in own_conds),
            attacker_side_aurora_veil=any(s.name == "AURORA_VEIL" for s in own_conds),
        )

    # ── per-Pokemon encoding ───────────────────────────────────────────────

    def _encode_own_pokemon(self, mon, battle, is_active: bool) -> np.ndarray:
        """Encode own Pokemon where all info is known exactly."""
        v = np.zeros(POKEMON_FEATURES, dtype=np.float32)

        v[0] = self._species_idx(mon.species)
        v[1] = float(mon.current_hp_fraction)
        v[2] = 1.0 if is_active else 0.0

        status_name = mon.status.name if mon.status else "none"
        v[3 + STATUS_TO_IDX.get(status_name, 0)] = 1.0

        for i, stat in enumerate(("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")):
            v[10 + i] = float(mon.boosts.get(stat, 0)) / 6.0

        fs = self._forme_manager.effective_base_stats("own", mon.species, mon.base_stats)
        for i, stat in enumerate(("hp", "atk", "def", "spa", "spd", "spe")):
            v[17 + i] = float(fs.get(stat, mon.base_stats.get(stat, 0))) / 255.0

        spe_base = fs.get("spe", mon.base_stats.get("spe", 0))
        v[23] = float(spe_base) / 200.0

        for move in mon.moves.values():
            tname = move.type.name if hasattr(move.type, "name") else str(move.type)
            if tname in TYPE_TO_IDX:
                v[24 + TYPE_TO_IDX[tname]] = 1.0

        for move in mon.moves.values():
            if getattr(move, "priority", 0) >= PRIORITY_THRESHOLD:
                v[42] = 1.0
                break

        for move in mon.moves.values():
            mid = to_id(move.id if hasattr(move, "id") else str(move))
            if mid in SETUP_MOVES:
                v[43] = 1.0
            if mid in HAZARD_MOVES:
                v[44] = 1.0
            if mid in HAZARD_REMOVAL_MOVES:
                v[45] = 1.0
            if mid in PIVOT_MOVES:
                v[46] = 1.0

        item_id = to_id(mon.item or "")
        for i, key_item in enumerate(KEY_ITEMS):
            v[47 + i] = 1.0 if item_id == key_item else 0.0

        v[55] = self._best_move_damage_norm(mon, battle)

        v[56] = 1.0 if not mon.is_terastallized else 0.0
        if mon.tera_type is not None:
            tname = mon.tera_type.name if hasattr(mon.tera_type, "name") else str(mon.tera_type)
            if tname in TYPE_TO_IDX:
                v[57 + TYPE_TO_IDX[tname]] = 1.0

        key = f"own:{to_id(mon.species)}"
        v[75] = min(self._times_active.get(key, 0), 10) / 10.0
        v[76] = 1.0  # own Pokemon always revealed

        return v

    def _encode_opp_pokemon(self, mon, is_active: bool) -> np.ndarray:
        """Encode opponent Pokemon using SetPredictor distribution for unknowns."""
        v = np.zeros(POKEMON_FEATURES, dtype=np.float32)
        revealed = mon.revealed if hasattr(mon, "revealed") else (mon.species != "")
        v[76] = 1.0 if revealed else 0.0

        if not revealed:
            return v

        self._update_predictor_from_pokemon(mon)
        pred = self._get_predictor(mon.species)

        v[0] = self._species_idx(mon.species)
        v[1] = float(mon.current_hp_fraction)
        v[2] = 1.0 if is_active else 0.0

        status_name = mon.status.name if mon.status else "none"
        v[3 + STATUS_TO_IDX.get(status_name, 0)] = 1.0

        for i, stat in enumerate(("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")):
            v[10 + i] = float(mon.boosts.get(stat, 0)) / 6.0

        bs = mon.base_stats
        fs = self._forme_manager.effective_base_stats("opp", mon.species, bs)
        for i, stat in enumerate(("hp", "atk", "def", "spa", "spd", "spe")):
            v[17 + i] = float(fs.get(stat, bs.get(stat, 0))) / 255.0

        spe = fs.get("spe", bs.get("spe", 0))
        if pred is not None:
            item_exp = pred.expected_attr("items")
            scarf_prob = sum(fq for nm, fq in item_exp.items() if to_id(nm) == "choicescarf")
            spe_eff = spe * (1.0 + 0.5 * scarf_prob)
        else:
            spe_eff = float(spe)
        v[23] = min(spe_eff / 200.0, 2.0)

        if pred is not None:
            type_probs = pred.expected_move_type_probs(self._move_db)
            for type_name, prob in type_probs.items():
                tname_up = type_name.upper()
                if tname_up in TYPE_TO_IDX:
                    v[24 + TYPE_TO_IDX[tname_up]] = float(prob)

        if pred is not None:
            priority_ids = {to_id(k) for k, mv in self._move_db.items()
                            if mv.get("priority", 0) >= PRIORITY_THRESHOLD}
            v[42] = pred.prob_has_flag(priority_ids, self._move_db)
            v[43] = pred.prob_has_flag(SETUP_MOVES, self._move_db)
            v[44] = pred.prob_has_flag(HAZARD_MOVES, self._move_db)
            v[45] = pred.prob_has_flag(HAZARD_REMOVAL_MOVES, self._move_db)
            v[46] = pred.prob_has_flag(PIVOT_MOVES, self._move_db)

        if pred is not None:
            item_exp = pred.expected_attr("items")
            for i, key_item in enumerate(KEY_ITEMS):
                v[47 + i] = float(
                    sum(fq for nm, fq in item_exp.items() if to_id(nm) == key_item)
                )

        v[55] = 0.5  # opponent's expected incoming damage unknown — placeholder midpoint

        v[56] = 0.0 if mon.is_terastallized else 1.0
        if mon.is_terastallized and mon.tera_type is not None:
            tname = mon.tera_type.name if hasattr(mon.tera_type, "name") else str(mon.tera_type)
            if tname in TYPE_TO_IDX:
                v[57 + TYPE_TO_IDX[tname]] = 1.0
        elif pred is not None:
            tera_exp = pred.expected_attr("teraTypes")
            for tname, prob in tera_exp.items():
                tup = tname.upper()
                if tup in TYPE_TO_IDX:
                    v[57 + TYPE_TO_IDX[tup]] += float(prob)

        key = f"opp:{to_id(mon.species)}"
        v[75] = min(self._times_active.get(key, 0), 10) / 10.0

        return v

    def _best_move_damage_norm(self, mon, battle) -> float:
        """
        Estimate best move expected damage against the active opponent,
        returned as fraction of the opponent's inferred max HP.

        Routes through DamageCalculator so item/ability/weather multipliers
        are applied consistently. See module docstring for why this is
        intentionally NOT unified with the opponent's damage placeholder at [55].
        """
        if not mon.moves:
            return 0.0
        opp = battle.opponent_active_pokemon
        if opp is None:
            return 0.0

        # Build attacker PokemonState from poke-env object
        mon_types = [t.name for t in mon.types if t is not None]
        atk_state = PokemonState(
            species=mon.species,
            level=mon.level or 50,
            base_stats=dict(mon.base_stats),
            boosts=dict(mon.boosts),
            ability=to_id(mon.ability) if mon.ability else None,
            item=to_id(mon.item or ""),
            types=mon_types,
            status=mon.status.name if mon.status else None,
        )

        # Build defender PokemonState
        opp_types = [t.name for t in opp.types if t is not None]
        def_state = PokemonState(
            species=opp.species,
            level=opp.level or 50,
            base_stats=dict(opp.base_stats),
            boosts=dict(opp.boosts),
            ability=to_id(opp.ability) if opp.ability else None,
            item=to_id(opp.item or ""),
            types=opp_types,
        )

        field = self._build_field_state(battle)

        # Infer opponent's max HP for normalization (same stat formula as DamageCalculator)
        opp_max_hp = _infer_stat(
            opp.base_stats.get("hp", 45), opp.level or 50, "hp"
        )
        if opp_max_hp <= 0:
            return 0.0

        best_frac = 0.0
        for move in mon.moves.values():
            # poke-env category names are uppercase; DamageCalculator expects title-case
            cat = move.category.name.title() if hasattr(move, "category") and move.category else "Physical"
            if cat == "Status":
                continue
            bp = getattr(move, "base_power", 0)
            if bp == 0:
                continue
            # poke-env type names are uppercase (e.g. "ELECTRIC") — _stab() normalises internally
            move_type = move.type.name if hasattr(move.type, "name") else "NORMAL"
            m_state = MoveState(
                move_id=to_id(move.id if hasattr(move, "id") else ""),
                base_power=bp,
                move_type=move_type,
                category=cat,
            )
            try:
                lo, hi = self._calc.calculate(atk_state, m_state, def_state, field)
                mean_dmg = (lo + hi) / 2.0
                best_frac = max(best_frac, mean_dmg / opp_max_hp)
            except Exception:
                pass

        return min(best_frac, 2.0)

    # ── global field encoding ──────────────────────────────────────────────

    def _encode_field(self, battle) -> np.ndarray:
        v = np.zeros(GLOBAL_FEATURES, dtype=np.float32)

        # Weather
        weather_label = "none"
        for w in battle.weather:
            weather_label = WEATHER_MAP.get(w.name, "none")
            break
        idx = WEATHER_ORDER.index(weather_label) if weather_label in WEATHER_ORDER else 0
        v[idx] = 1.0

        # Terrain
        terrain_label = "none"
        for f in battle.fields:
            label = TERRAIN_MAP.get(f.name, "")
            if label:
                terrain_label = label
                break
        tidx = TERRAIN_ORDER.index(terrain_label) if terrain_label in TERRAIN_ORDER else 0
        v[5 + tidx] = 1.0

        # Trick Room — activation_turn stored as field value; remaining = 5 - elapsed
        trick_active = any(f.name == "TRICK_ROOM" for f in battle.fields)
        v[10] = 1.0 if trick_active else 0.0
        if trick_active:
            activation_turn = next(
                (t for f, t in battle.fields.items() if f.name == "TRICK_ROOM"),
                battle.turn,  # safe default: assume just activated this turn
            )
            turns_elapsed = max(0, battle.turn - activation_turn)
            remaining = max(0, 5 - turns_elapsed)
            v[11] = remaining / 5.0

        # Hazards
        def hazard_vec(side_conds):
            hv = np.zeros(4, dtype=np.float32)
            hv[0] = 1.0 if any(s.name == "STEALTH_ROCK" for s in side_conds) else 0.0
            spikes = next((cnt for s, cnt in side_conds.items() if s.name == "SPIKES"), 0)
            hv[1] = spikes / 3.0
            tspikes = next((cnt for s, cnt in side_conds.items() if s.name == "TOXIC_SPIKES"), 0)
            hv[2] = tspikes / 2.0
            hv[3] = 1.0 if any(s.name == "STICKY_WEB" for s in side_conds) else 0.0
            return hv

        v[12:16] = hazard_vec(battle.side_conditions)
        v[16:20] = hazard_vec(battle.opponent_side_conditions)

        # Screens — activation_turn stored in side_conditions dict value
        def screen_vec(side_conds, turn):
            sv = np.zeros(6, dtype=np.float32)
            for name, offset, duration in (
                ("REFLECT", 0, 5),
                ("LIGHT_SCREEN", 2, 5),
                ("AURORA_VEIL", 4, 8),  # 8 with Light Clay (common in randbats)
            ):
                active = any(s.name == name for s in side_conds)
                sv[offset] = 1.0 if active else 0.0
                if active:
                    act_turn = next(
                        (t for s, t in side_conds.items() if s.name == name),
                        turn,  # safe default: just activated
                    )
                    turns_elapsed = max(0, turn - act_turn)
                    rem = max(0, duration - turns_elapsed)
                    sv[offset + 1] = rem / duration
            return sv

        v[20:26] = screen_vec(battle.side_conditions, battle.turn)
        v[26:32] = screen_vec(battle.opponent_side_conditions, battle.turn)

        v[32] = min(battle.turn / 100.0, 1.0)

        our_rem = sum(1 for m in battle.team.values() if not m.fainted)
        v[33] = our_rem / 6.0

        # opp_remaining: fainted Pokemon are always in opponent_team, so this is correct
        # even when some opponents are unrevealed (they add to remaining by not being fainted).
        opp_rem = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
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
            if is_transitional(opp_active.species):
                self._forme_manager.get("opp", opp_active.species).on_damage_taken(
                    opp_active.current_hp_fraction
                )

    # ── main extraction ────────────────────────────────────────────────────

    def extract(self, battle) -> np.ndarray:
        """
        Extract the full feature vector from a poke-env Battle object.
        Returns float32 array of shape (FEATURE_DIM,) = (959,).
        """
        t0 = time.perf_counter()

        # Initialise / update slot orders
        if self._own_slot_order is None:
            self._init_own_slots(battle)
        self._update_opp_slots(battle)
        self._update_activity(battle)

        vec = np.zeros(FEATURE_DIM, dtype=np.float32)

        # Determine active species for is_active encoding
        own_active_sid = to_id(battle.active_pokemon.species) if battle.active_pokemon else None
        opp_active_sid = (to_id(battle.opponent_active_pokemon.species)
                          if battle.opponent_active_pokemon else None)

        # Own team (slots 0-5) — stable team-slot order
        own_by_sid = {to_id(m.species): m for m in battle.team.values()}
        for slot, sid in enumerate(self._own_slot_order[:TEAM_SIZE]):
            mon = own_by_sid.get(sid)
            start = slot * POKEMON_FEATURES
            if mon is not None and not mon.fainted:
                is_active = (sid == own_active_sid)
                vec[start:start + POKEMON_FEATURES] = self._encode_own_pokemon(mon, battle, is_active)
            # fainted or unresolved slot stays zero-filled

        # Opponent team (slots 6-11) — stable reveal order
        opp_by_sid = {to_id(m.species): m for m in battle.opponent_team.values()}
        for slot, sid in enumerate(self._opp_slot_order[:TEAM_SIZE]):
            mon = opp_by_sid.get(sid)
            start = (TEAM_SIZE + slot) * POKEMON_FEATURES
            if mon is not None:
                is_active = (sid == opp_active_sid)
                vec[start:start + POKEMON_FEATURES] = self._encode_opp_pokemon(mon, is_active)
            # unrevealed slot stays zero-filled

        # Global field
        vec[TEAM_SIZE * 2 * POKEMON_FEATURES:] = self._encode_field(battle)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 10.0:
            logger.warning("Feature extraction took %.1f ms (> 10 ms threshold)", elapsed_ms)

        if not self._feature_dim_verified:
            assert vec.shape == (FEATURE_DIM,), f"Shape mismatch: {vec.shape}"
            assert not np.any(np.isnan(vec)), "NaN in feature vector"
            assert not np.any(np.isinf(vec)), "Inf in feature vector"
            self._feature_dim_verified = True

        return vec
