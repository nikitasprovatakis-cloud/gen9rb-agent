"""
D4: Stateful in-battle forme tracking.

Handles Pokemon whose stats or move types change mid-battle:
  - Palafin:   Zero → Hero on switch out + back in
  - Morpeko:   Full Belly ↔ Hangry each turn (Aura Wheel type flips)
  - Minior:    Meteor → Core at ≤50% HP (irreversible)
  - Terapagos: Normal → Stellar on Tera activation
  - Cramorant: Gulping after Surf/Dive; Gorging if HP>50%; reverts on contact

Decision: current-forme stats only (not probability-weighted future formes).
Rationale: the network sees state sequences so it can learn transition dynamics
from training data. Probability-weighting adds complexity without clear benefit
at this stage.

For edge cases (e.g. Palafin fainting in Zero forme before returning),
we default to the last observable forme and log a warning.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Forme stat overrides ───────────────────────────────────────────────────
# Only species whose stats differ by forme. Source: Bulbapedia / smogon/randbats.
# Format: {species_id: {forme_id: {stat: value}}}

FORME_BASE_STATS: dict[str, dict[str, dict[str, int]]] = {
    "palafin": {
        "palafin-zero": {"hp": 100, "atk": 70, "def": 72, "spa": 53, "spd": 62, "spe": 100},
        "palafin-hero": {"hp": 100, "atk": 160, "def": 97, "spa": 106, "spd": 87, "spe": 100},
    },
    "minior": {
        "minior-meteor": {"hp": 60, "atk": 60, "def": 100, "spa": 60, "spd": 100, "spe": 60},
        "minior-core":   {"hp": 60, "atk": 100, "def": 60, "spa": 100, "spd": 60, "spe": 120},
    },
    "morpeko": {
        "morpeko":        {"hp": 58, "atk": 95, "def": 58, "spa": 70, "spd": 58, "spe": 97},
        "morpeko-hangry": {"hp": 58, "atk": 95, "def": 58, "spa": 70, "spd": 58, "spe": 97},
    },
    "terapagos": {
        "terapagos":         {"hp": 90, "atk": 65, "def": 85, "spa": 65, "spd": 85, "spe": 60},
        "terapagos-terastal":{"hp": 90, "atk": 65, "def": 85, "spa": 65, "spd": 85, "spe": 60},
        "terapagos-stellar": {"hp": 160, "atk": 105, "def": 110, "spa": 130, "spd": 110, "spe": 85},
    },
    "cramorant": {
        "cramorant":         {"hp": 70, "atk": 65, "def": 45, "spa": 75, "spd": 45, "spe": 85},
        "cramorant-gulping": {"hp": 70, "atk": 65, "def": 45, "spa": 75, "spd": 45, "spe": 85},
        "cramorant-gorging": {"hp": 70, "atk": 65, "def": 45, "spa": 75, "spd": 45, "spe": 85},
    },
}

# Species whose Aura Wheel move type changes by forme
MORPEKO_AURA_WHEEL_TYPE: dict[str, str] = {
    "morpeko":        "Electric",
    "morpeko-hangry": "Dark",
}


class FormeTracker:
    """
    Tracks in-battle forme state for one Pokemon (identified by species + slot).

    Usage:
        tracker = FormeTracker(species="Palafin")
        tracker.on_switch_out()   # goes Hero
        tracker.on_switch_in()    # back in as Hero
        forme = tracker.current_forme   # "palafin-hero"
        stats = tracker.effective_base_stats  # Hero's stats
    """

    # Default starting forme per species (where it differs from the raw species ID)
    _INITIAL_FORME: dict[str, str] = {
        "palafin": "palafin-zero",
        "minior":  "minior-meteor",
    }

    def __init__(self, species: str):
        from knowledge.set_pool import to_id
        self._species_id = to_id(species)
        self._current_forme: str = self._INITIAL_FORME.get(self._species_id, self._species_id)
        self._switch_count: int = 0  # how many times has this Pokemon switched out
        self._fainted: bool = False

    # ── event hooks ───────────────────────────────────────────────────────

    def on_switch_out(self) -> None:
        """Called when this Pokemon leaves the field."""
        if self._fainted:
            return
        if self._species_id == "palafin":
            self._switch_count += 1
            # Becomes Hero after first switch-out
            if self._switch_count >= 1:
                self._current_forme = "palafin-hero"

    def on_switch_in(self) -> None:
        """Called when this Pokemon enters the field."""
        pass  # forme already set during switch_out for Palafin

    def on_damage_taken(self, hp_fraction: float) -> None:
        """Called when HP changes. hp_fraction is current HP / max HP."""
        if self._species_id == "minior":
            if hp_fraction <= 0.5 and "meteor" in self._current_forme:
                self._current_forme = "minior-core"
                logger.debug("Minior transitioned to Core forme (HP %.0f%%)", hp_fraction * 100)
        if hp_fraction <= 0.0:
            self._fainted = True

    def on_turn_end(self, turn: int) -> None:
        """Called at end of each turn. Morpeko alternates each turn."""
        if self._species_id == "morpeko":
            if turn % 2 == 1:
                self._current_forme = "morpeko-hangry"
            else:
                self._current_forme = "morpeko"

    def on_tera(self) -> None:
        """Called when this Pokemon Terastallizes."""
        if self._species_id == "terapagos":
            self._current_forme = "terapagos-stellar"

    def on_contact_taken(self) -> None:
        """Called when Cramorant takes a contact hit."""
        if self._species_id == "cramorant":
            self._current_forme = "cramorant"

    def on_use_move(self, move_id: str) -> None:
        """Called when Cramorant uses Surf or Dive."""
        if self._species_id == "cramorant":
            from knowledge.set_pool import to_id
            if to_id(move_id) in ("surf", "dive"):
                self._current_forme = "cramorant-gulping"

    # ── queries ───────────────────────────────────────────────────────────

    @property
    def current_forme(self) -> str:
        return self._current_forme

    @property
    def effective_base_stats(self) -> Optional[dict[str, int]]:
        """Return base stats for current forme, or None if not a transitional species."""
        species_formes = FORME_BASE_STATS.get(self._species_id)
        if species_formes is None:
            return None
        return species_formes.get(self._current_forme)

    def aura_wheel_type(self) -> Optional[str]:
        """Return Aura Wheel move type for current Morpeko forme, or None."""
        return MORPEKO_AURA_WHEEL_TYPE.get(self._current_forme)


# ── Battle-level forme manager ─────────────────────────────────────────────

class FormeManager:
    """
    Manages FormeTracker instances for all Pokemon in a battle (own + opponent).
    Keyed by (side, slot_index) or (side, species_id).
    """

    def __init__(self):
        self._trackers: dict[str, FormeTracker] = {}

    def _key(self, side: str, species: str) -> str:
        from knowledge.set_pool import to_id
        return f"{side}:{to_id(species)}"

    def get(self, side: str, species: str) -> FormeTracker:
        key = self._key(side, species)
        if key not in self._trackers:
            self._trackers[key] = FormeTracker(species)
        return self._trackers[key]

    def reset(self) -> None:
        self._trackers.clear()

    def effective_base_stats(self, side: str, species: str,
                              fallback: dict) -> dict[str, int]:
        """Return base stats accounting for current forme, falling back to poke-env's stats."""
        tracker = self.get(side, species)
        override = tracker.effective_base_stats
        return override if override is not None else fallback


# ── Utility: identify transitional species from randbats pool ──────────────

_TRANSITIONAL_SPECIES = frozenset(FORME_BASE_STATS.keys())


def is_transitional(species: str) -> bool:
    """Return True if this species has in-battle forme changes we track."""
    from knowledge.set_pool import to_id
    return to_id(species) in _TRANSITIONAL_SPECIES
