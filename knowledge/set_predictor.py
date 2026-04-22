"""
D2: Bayesian set predictor for Gen 9 Random Battle opponents.

pkmn/randbats data is role-based: each species has 1-N roles
(e.g. "Setup Sweeper", "Fast Support") with empirical role weights and
per-role move/item/ability/tera-type frequencies.

We model uncertainty as a distribution over roles. Observing a move, item,
ability, or tera type performs a Bayesian update:

  P(role | observed) ∝ P(role) * P(observed | role)

where P(observed | role) = the empirical frequency of that attribute in that
role's data (0.0 if not present → eliminates the role).

No special-case handling for Choice items — behavioral inference from the
feature vector is left to the policy network.
"""

import logging
import re
from typing import Optional

from knowledge.set_pool import get_species_data, to_id

logger = logging.getLogger(__name__)


class SetPredictor:
    """
    Bayesian updater over pkmn/randbats role distribution for one opponent Pokemon.

    Usage:
        pred = SetPredictor("Garchomp")
        pred.observe_move("Earthquake")
        pred.observe_move("Swords Dance")
        pred.observe_item("Choice Scarf")
        dist = pred.get_distribution()   # [(role_name, role_data, probability), ...]
    """

    def __init__(self, species: str, cache_dir=None):
        self.species = species
        data = get_species_data(species, cache_dir)
        self._level: int = data["level"]
        self._roles: dict = data["roles"]

        # Normalize role weights into a proper probability distribution
        total = sum(r.get("weight", 1.0) for r in self._roles.values())
        if total < 1e-9:
            total = len(self._roles)
            self._probs = {r: 1.0 / total for r in self._roles}
        else:
            self._probs = {r: d.get("weight", 1.0) / total
                          for r, d in self._roles.items()}

        self._prior: dict[str, float] = dict(self._probs)

    # ── internal ──────────────────────────────────────────────────────────

    def _attr_likelihoods(self, attr: str, value_id: str) -> dict[str, float]:
        """
        P(attr = value | role) for each role. Matched by Showdown ID comparison.
        Returns 0.0 for roles that don't have this attribute at all.
        """
        likelihoods = {}
        for role_name, role_data in self._roles.items():
            freq = 0.0
            for key, f in role_data.get(attr, {}).items():
                if to_id(key) == value_id:
                    freq = float(f)
                    break
            likelihoods[role_name] = freq
        return likelihoods

    def _update(self, likelihoods: dict[str, float]) -> None:
        """Multiply current probs by likelihoods, renormalize. Keeps prior on near-zero mass."""
        new_probs = {role: self._probs[role] * likelihoods.get(role, 0.0)
                     for role in self._probs}
        total = sum(new_probs.values())
        if total < 1e-9:
            logger.warning(
                "SetPredictor(%s): observation eliminated all roles — keeping prior",
                self.species,
            )
            return
        self._probs = {role: p / total for role, p in new_probs.items()}

    # ── observe API ───────────────────────────────────────────────────────

    def observe_move(self, move: str) -> None:
        """Update distribution after observing this Pokemon use a move."""
        self._update(self._attr_likelihoods("moves", to_id(move)))

    def observe_item(self, item: str) -> None:
        """Update distribution after the item is revealed."""
        self._update(self._attr_likelihoods("items", to_id(item)))

    def observe_ability(self, ability: str) -> None:
        """Update distribution after the ability is revealed."""
        self._update(self._attr_likelihoods("abilities", to_id(ability)))

    def observe_tera(self, tera_type: str) -> None:
        """Update distribution after Tera type is revealed."""
        self._update(self._attr_likelihoods("teraTypes", to_id(tera_type)))

    def observe_stat_boost(self, stat: str) -> None:
        pass  # No set-level inference from boosts yet

    def observe_status(self, status: str) -> None:
        pass  # No set-level inference from status yet

    # ── query API ─────────────────────────────────────────────────────────

    def get_distribution(self) -> list[tuple[str, dict, float]]:
        """
        Return sorted list of (role_name, role_data_dict, probability), highest first.
        role_data_dict is the raw pkmn/randbats role entry for reference.
        """
        return sorted(
            [(role, self._roles[role], prob) for role, prob in self._probs.items()],
            key=lambda x: x[2],
            reverse=True,
        )

    def top_role(self) -> tuple[str, dict, float]:
        """Return the (role_name, role_data, probability) of the highest-probability role."""
        return self.get_distribution()[0]

    def expected_attr(self, attr: str) -> dict[str, float]:
        """
        Compute P(attr = value) marginalized over current role distribution.

        For attr in {"moves", "items", "abilities", "teraTypes"}.
        Returns {canonical_name: probability} for all values seen across roles.
        """
        result: dict[str, float] = {}
        for role_name, role_data, role_prob in self.get_distribution():
            for val, freq in role_data.get(attr, {}).items():
                result[val] = result.get(val, 0.0) + role_prob * float(freq)
        return result

    def prob_has_move(self, move: str) -> float:
        """P(this Pokemon has move) under current distribution."""
        move_id = to_id(move)
        total = 0.0
        for role_name, role_data, role_prob in self.get_distribution():
            for m, freq in role_data.get("moves", {}).items():
                if to_id(m) == move_id:
                    total += role_prob * float(freq)
                    break
        return total

    def expected_move_type_probs(self, move_db: dict) -> dict[str, float]:
        """
        Returns {type_name: probability} that this Pokemon has at least one move of that type.

        move_db: dict of {move_id: move_data_dict} from poke-env GenData.
        Uses P(type T present) ≈ 1 - prod(1 - P(move_i of type T)) approximation.
        """
        # Collect P(has each move) under current distribution
        move_probs: dict[str, float] = {}
        for role_name, role_data, role_prob in self.get_distribution():
            for move_name, freq in role_data.get("moves", {}).items():
                mid = to_id(move_name)
                move_probs[mid] = move_probs.get(mid, 0.0) + role_prob * float(freq)

        # Aggregate by type
        type_no_move_prob: dict[str, float] = {}  # P(no move of this type)
        for move_id, prob in move_probs.items():
            mdata = move_db.get(move_id, {})
            mtype = mdata.get("type", "")
            if not mtype:
                continue
            # P(not having this move) contribution: independence approximation
            type_no_move_prob[mtype] = type_no_move_prob.get(mtype, 1.0) * (1.0 - prob)

        return {t: 1.0 - no_p for t, no_p in type_no_move_prob.items()}

    def prob_has_flag(self, flag_move_ids: set, move_db: dict) -> float:
        """P(has at least one move from flag_move_ids set) under current distribution."""
        move_probs: dict[str, float] = {}
        for role_name, role_data, role_prob in self.get_distribution():
            for move_name, freq in role_data.get("moves", {}).items():
                mid = to_id(move_name)
                move_probs[mid] = move_probs.get(mid, 0.0) + role_prob * float(freq)

        prob_none = 1.0
        for mid in flag_move_ids:
            if mid in move_probs:
                prob_none *= (1.0 - move_probs[mid])
        return 1.0 - prob_none

    @property
    def level(self) -> int:
        return self._level

    def reset(self) -> None:
        """Reset to prior distribution (start of battle)."""
        self._probs = dict(self._prior)
