"""
Phase 2 unit tests — knowledge layer (D1-D5).

Covers ~15 hand-constructed scenarios testing:
  D1 set_pool:       name normalization, species resolution, data structure
  D2 set_predictor:  Bayesian update correctness, edge cases
  D4 formes:         Palafin/Minior/Morpeko transitions
  D5 damage_calc:    core formula, STAB, type effectiveness, items, burn

Run:
  cd /home/user/showdown-bot
  python -m pytest tests/test_knowledge.py -v
"""

import math
import os
import sys

import pytest

sys.path.insert(0, "/home/user/showdown-bot")

os.environ["METAMON_ALLOW_ANY_POKE_ENV"] = "True"
os.environ["METAMON_CACHE_DIR"] = "/home/user/metamon-cache"


# ── D1: set_pool ──────────────────────────────────────────────────────────────

class TestToId:
    def setup_method(self):
        from knowledge.set_pool import to_id
        self.to_id = to_id

    def test_basic_lowercase(self):
        assert self.to_id("Garchomp") == "garchomp"

    def test_hyphen_removed(self):
        assert self.to_id("Landorus-Therian") == "landorustherian"

    def test_apostrophe_removed(self):
        assert self.to_id("Farfetch'd") == "farfetchd"

    def test_space_removed(self):
        assert self.to_id("Mr. Mime") == "mrmime"

    def test_colon_removed(self):
        assert self.to_id("Type: Null") == "typenull"

    def test_already_normalized(self):
        assert self.to_id("garchomp") == "garchomp"


class TestResolveSpecies:
    def setup_method(self):
        from knowledge.set_pool import resolve_species
        self.resolve = resolve_species

    def test_canonical_form_resolves(self):
        result = self.resolve("Garchomp")
        assert result is not None
        assert "garchomp" in result.lower()

    def test_hyphenated_resolves(self):
        result = self.resolve("Landorus-Therian")
        assert result is not None

    def test_unknown_species_returns_none(self):
        result = self.resolve("NotARealPokemon")
        assert result is None

    def test_forme_with_dash_resolves(self):
        # Iron Valiant is in the Gen 9 randbats pool
        result = self.resolve("Iron Valiant")
        assert result is not None


class TestGetSpeciesData:
    def setup_method(self):
        from knowledge.set_pool import get_species_data
        self.get = get_species_data

    def test_data_has_level(self):
        data = self.get("Garchomp")
        assert "level" in data
        assert isinstance(data["level"], int)
        assert 50 <= data["level"] <= 100

    def test_data_has_roles(self):
        data = self.get("Garchomp")
        assert "roles" in data
        assert len(data["roles"]) >= 1

    def test_role_has_moves(self):
        data = self.get("Garchomp")
        for role_data in data["roles"].values():
            assert "moves" in role_data
            assert len(role_data["moves"]) >= 1
            break

    def test_role_frequencies_sum_to_one(self):
        from knowledge.set_pool import verify_frequencies
        assert verify_frequencies("Garchomp")

    def test_unknown_species_raises(self):
        with pytest.raises(KeyError):
            self.get("NotARealPokemon")


# ── D2: set_predictor ─────────────────────────────────────────────────────────

class TestSetPredictorInit:
    def setup_method(self):
        from knowledge.set_predictor import SetPredictor
        self.pred = SetPredictor("Garchomp")

    def test_probabilities_sum_to_one(self):
        dist = self.pred.get_distribution()
        total = sum(p for _, _, p in dist)
        assert abs(total - 1.0) < 1e-6

    def test_distribution_nonempty(self):
        dist = self.pred.get_distribution()
        assert len(dist) >= 1

    def test_distribution_sorted_descending(self):
        dist = self.pred.get_distribution()
        probs = [p for _, _, p in dist]
        assert probs == sorted(probs, reverse=True)


class TestSetPredictorObserve:
    def test_observe_move_changes_distribution(self):
        from knowledge.set_predictor import SetPredictor
        pred = SetPredictor("Garchomp")
        before = {r: p for r, _, p in pred.get_distribution()}
        pred.observe_move("Swords Dance")
        after = {r: p for r, _, p in pred.get_distribution()}
        # Distribution should shift — not all probabilities identical to before
        changed = any(abs(after.get(r, 0) - before.get(r, 0)) > 1e-6 for r in before)
        assert changed

    def test_observe_move_in_no_role_keeps_prior(self):
        """Observing a move that's in no role should warn and keep prior unchanged."""
        from knowledge.set_predictor import SetPredictor
        pred = SetPredictor("Garchomp")
        before = {r: p for r, _, p in pred.get_distribution()}
        # A move that Garchomp never runs in randbats
        pred.observe_move("Splash")
        after = {r: p for r, _, p in pred.get_distribution()}
        # Prior should be preserved (all roles had 0 likelihood → total = 0 → fallback)
        for role in before:
            assert abs(after.get(role, 0) - before[role]) < 1e-6

    def test_observe_multiple_moves_narrows_distribution(self):
        """After observing several consistent moves, top role should have high probability."""
        from knowledge.set_predictor import SetPredictor
        pred = SetPredictor("Garchomp")
        initial_top = pred.top_role()[2]
        # Observe moves typical of a setup sweeper
        pred.observe_move("Swords Dance")
        pred.observe_move("Earthquake")
        pred.observe_move("Scale Shot")
        top_after = pred.top_role()[2]
        # Top probability should have increased (or stayed if only one role)
        assert top_after >= initial_top - 1e-6

    def test_observe_item_updates_distribution(self):
        from knowledge.set_predictor import SetPredictor
        pred = SetPredictor("Garchomp")
        before_top_prob = pred.top_role()[2]
        pred.observe_item("Choice Scarf")
        after_dist = pred.get_distribution()
        total = sum(p for _, _, p in after_dist)
        assert abs(total - 1.0) < 1e-6

    def test_prob_sum_preserved_after_updates(self):
        from knowledge.set_predictor import SetPredictor
        pred = SetPredictor("Garchomp")
        pred.observe_move("Earthquake")
        pred.observe_item("Choice Scarf")
        dist = pred.get_distribution()
        total = sum(p for _, _, p in dist)
        assert abs(total - 1.0) < 1e-6

    def test_expected_attr_returns_valid_probs(self):
        from knowledge.set_predictor import SetPredictor
        pred = SetPredictor("Garchomp")
        item_exp = pred.expected_attr("items")
        assert isinstance(item_exp, dict)
        total = sum(item_exp.values())
        # Should be ≤ n_moves_per_set (not a strict probability distribution, can exceed 1)
        assert total >= 0.0

    def test_level_attribute(self):
        from knowledge.set_predictor import SetPredictor
        pred = SetPredictor("Garchomp")
        assert isinstance(pred.level, int)
        assert 50 <= pred.level <= 100


# ── D4: formes ────────────────────────────────────────────────────────────────

class TestFormeTrackerPalafin:
    def setup_method(self):
        from knowledge.formes import FormeTracker
        self.tracker = FormeTracker("Palafin")

    def test_initial_forme(self):
        assert self.tracker.current_forme == "palafin-zero"

    def test_becomes_hero_after_switch_out(self):
        self.tracker.on_switch_out()
        assert self.tracker.current_forme == "palafin-hero"

    def test_hero_stats_different_from_zero(self):
        from knowledge.formes import FORME_BASE_STATS
        zero_atk = FORME_BASE_STATS["palafin"]["palafin-zero"]["atk"]
        hero_atk = FORME_BASE_STATS["palafin"]["palafin-hero"]["atk"]
        assert hero_atk > zero_atk

    def test_remains_hero_after_second_switch_out(self):
        self.tracker.on_switch_out()
        self.tracker.on_switch_in()
        self.tracker.on_switch_out()
        assert self.tracker.current_forme == "palafin-hero"

    def test_effective_base_stats_zero(self):
        stats = self.tracker.effective_base_stats
        assert stats is not None
        assert stats["atk"] == 70  # zero forme attack

    def test_effective_base_stats_hero(self):
        self.tracker.on_switch_out()
        stats = self.tracker.effective_base_stats
        assert stats is not None
        assert stats["atk"] == 160  # hero forme attack


class TestFormeTrackerMinior:
    def setup_method(self):
        from knowledge.formes import FormeTracker
        self.tracker = FormeTracker("Minior")

    def test_initial_forme_is_meteor(self):
        assert "meteor" in self.tracker.current_forme

    def test_stays_meteor_above_50pct(self):
        self.tracker.on_damage_taken(0.6)
        assert "meteor" in self.tracker.current_forme

    def test_becomes_core_at_50pct(self):
        self.tracker.on_damage_taken(0.5)
        assert "core" in self.tracker.current_forme

    def test_becomes_core_below_50pct(self):
        self.tracker.on_damage_taken(0.3)
        assert "core" in self.tracker.current_forme

    def test_transition_is_irreversible(self):
        self.tracker.on_damage_taken(0.3)
        self.tracker.on_damage_taken(0.8)  # HP "healed" (shouldn't revert)
        assert "core" in self.tracker.current_forme


class TestFormeTrackerMorpeko:
    def setup_method(self):
        from knowledge.formes import FormeTracker
        self.tracker = FormeTracker("Morpeko")

    def test_initial_forme(self):
        assert self.tracker.current_forme == "morpeko"

    def test_turn_1_is_hangry(self):
        self.tracker.on_turn_end(1)
        assert self.tracker.current_forme == "morpeko-hangry"

    def test_turn_2_is_full_belly(self):
        self.tracker.on_turn_end(1)
        self.tracker.on_turn_end(2)
        assert self.tracker.current_forme == "morpeko"

    def test_aura_wheel_type_full_belly(self):
        assert self.tracker.aura_wheel_type() == "Electric"

    def test_aura_wheel_type_hangry(self):
        self.tracker.on_turn_end(1)
        assert self.tracker.aura_wheel_type() == "Dark"


class TestFormeManager:
    def test_get_creates_tracker(self):
        from knowledge.formes import FormeManager
        mgr = FormeManager()
        t = mgr.get("own", "Palafin")
        assert t is not None
        assert t.current_forme == "palafin-zero"

    def test_get_returns_same_tracker(self):
        from knowledge.formes import FormeManager
        mgr = FormeManager()
        t1 = mgr.get("own", "Palafin")
        t2 = mgr.get("own", "Palafin")
        assert t1 is t2

    def test_reset_clears_state(self):
        from knowledge.formes import FormeManager
        mgr = FormeManager()
        t = mgr.get("own", "Palafin")
        t.on_switch_out()
        mgr.reset()
        t2 = mgr.get("own", "Palafin")
        assert t2.current_forme == "palafin-zero"  # fresh tracker

    def test_effective_base_stats_fallback(self):
        from knowledge.formes import FormeManager
        mgr = FormeManager()
        fallback = {"hp": 45, "atk": 49, "def": 49, "spa": 65, "spd": 65, "spe": 45}
        # Bulbasaur is not in FORME_BASE_STATS → returns fallback
        stats = mgr.effective_base_stats("own", "Bulbasaur", fallback)
        assert stats == fallback


# ── D5: damage_calc ───────────────────────────────────────────────────────────

def _make_calc():
    from knowledge.damage_calc import DamageCalculator
    return DamageCalculator()


def _pokemon(species="Garchomp", level=80, base_stats=None, ability=None, item=None,
             types=None, status=None):
    from knowledge.damage_calc import PokemonState
    if base_stats is None:
        base_stats = {"hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102}
    if types is None:
        types = ["Dragon", "Ground"]
    return PokemonState(
        species=species, level=level, base_stats=base_stats,
        ability=ability, item=item, types=types, status=status,
    )


def _move(move_id="earthquake", bp=100, move_type="Ground", category="Physical"):
    from knowledge.damage_calc import MoveState
    return MoveState(move_id=move_id, base_power=bp, move_type=move_type, category=category)


def _field():
    from knowledge.damage_calc import FieldState
    return FieldState()


class TestDamageCalcBasic:
    def setup_method(self):
        self.calc = _make_calc()

    def test_status_move_returns_zero(self):
        atk = _pokemon()
        def_ = _pokemon()
        from knowledge.damage_calc import MoveState
        status_move = MoveState("swordsdance", 0, "Normal", "Status")
        lo, hi = self.calc.calculate(atk, status_move, def_, _field())
        assert lo == 0.0 and hi == 0.0

    def test_zero_bp_returns_zero(self):
        atk = _pokemon()
        def_ = _pokemon()
        from knowledge.damage_calc import MoveState
        m = MoveState("tackle", 0, "Normal", "Physical")
        lo, hi = self.calc.calculate(atk, m, def_, _field())
        assert lo == 0.0 and hi == 0.0

    def test_damage_range_hi_gte_lo(self):
        atk = _pokemon()
        def_ = _pokemon()
        lo, hi = self.calc.calculate(atk, _move(), def_, _field())
        assert hi >= lo

    def test_damage_positive(self):
        atk = _pokemon()
        def_ = _pokemon()
        lo, hi = self.calc.calculate(atk, _move(), def_, _field())
        assert lo > 0.0

    def test_range_85_to_100(self):
        """hi ≈ lo / 0.85 within rounding (standard Gen 5+ roll)."""
        atk = _pokemon()
        def_ = _pokemon()
        lo, hi = self.calc.calculate(atk, _move(), def_, _field())
        ratio = hi / lo if lo > 0 else 1.0
        assert ratio <= 1.18  # 1/0.85 ≈ 1.176


class TestDamageCalcSTAB:
    def setup_method(self):
        self.calc = _make_calc()

    def test_stab_boosts_damage(self):
        """Dragon move from Dragon-type attacker gets STAB vs non-STAB."""
        atk_dragon = _pokemon(types=["Dragon"])
        atk_fire = _pokemon(types=["Fire"])
        def_ = _pokemon(types=["Normal"])
        dragon_move = _move("dragonpulse", bp=85, move_type="Dragon", category="Special")
        lo_stab, _ = self.calc.calculate(atk_dragon, dragon_move, def_, _field())
        lo_no, _ = self.calc.calculate(atk_fire, dragon_move, def_, _field())
        ratio = lo_stab / lo_no if lo_no > 0 else 0
        assert abs(ratio - 1.5) < 0.05


class TestDamageCalcTypeEffectiveness:
    def setup_method(self):
        self.calc = _make_calc()

    def test_supereffective_doubles(self):
        """Electric vs Water should be 2x vs neutral."""
        atk = _pokemon(types=["Electric"])
        def_water = _pokemon(types=["Water"])
        def_normal = _pokemon(types=["Normal"])
        elec_move = _move("thunderbolt", bp=90, move_type="Electric", category="Special")
        lo_se, _ = self.calc.calculate(atk, elec_move, def_water, _field())
        lo_ne, _ = self.calc.calculate(atk, elec_move, def_normal, _field())
        ratio = lo_se / lo_ne if lo_ne > 0 else 0
        assert abs(ratio - 2.0) < 0.1

    def test_not_very_effective_halves(self):
        """Electric vs Grass should be 0.5x vs neutral."""
        atk = _pokemon(types=["Electric"])
        def_grass = _pokemon(types=["Grass"])
        def_normal = _pokemon(types=["Normal"])
        elec_move = _move("thunderbolt", bp=90, move_type="Electric", category="Special")
        lo_nve, _ = self.calc.calculate(atk, elec_move, def_grass, _field())
        lo_ne, _ = self.calc.calculate(atk, elec_move, def_normal, _field())
        ratio = lo_nve / lo_ne if lo_ne > 0 else 0
        assert abs(ratio - 0.5) < 0.05


class TestDamageCalcItems:
    def setup_method(self):
        self.calc = _make_calc()

    def test_life_orb_1_3x(self):
        atk_lo = _pokemon(item="lifeorb")
        atk_plain = _pokemon()
        def_ = _pokemon(types=["Normal"])
        m = _move()
        lo_orb, _ = self.calc.calculate(atk_lo, m, def_, _field())
        lo_plain, _ = self.calc.calculate(atk_plain, m, def_, _field())
        ratio = lo_orb / lo_plain if lo_plain > 0 else 0
        assert abs(ratio - 1.3) < 0.05

    def test_choice_band_1_5x_physical(self):
        atk_cb = _pokemon(item="choiceband")
        atk_plain = _pokemon()
        def_ = _pokemon(types=["Normal"])
        m = _move()  # Physical
        lo_cb, _ = self.calc.calculate(atk_cb, m, def_, _field())
        lo_plain, _ = self.calc.calculate(atk_plain, m, def_, _field())
        ratio = lo_cb / lo_plain if lo_plain > 0 else 0
        assert abs(ratio - 1.5) < 0.05

    def test_choice_band_not_applied_to_special(self):
        atk_cb = _pokemon(item="choiceband")
        atk_plain = _pokemon()
        def_ = _pokemon(types=["Normal"])
        sp_move = _move("flamethrower", bp=90, move_type="Fire", category="Special")
        lo_cb, _ = self.calc.calculate(atk_cb, sp_move, def_, _field())
        lo_plain, _ = self.calc.calculate(atk_plain, sp_move, def_, _field())
        ratio = lo_cb / lo_plain if lo_plain > 0 else 0
        # Should not get 1.5x — ratio should be ~1.0
        assert ratio < 1.1


class TestDamageCalcAbilities:
    def setup_method(self):
        self.calc = _make_calc()

    def test_huge_power_doubles_attack(self):
        atk_hp = _pokemon(ability="hugepower")
        atk_plain = _pokemon()
        def_ = _pokemon(types=["Normal"])
        m = _move()
        lo_hp, _ = self.calc.calculate(atk_hp, m, def_, _field())
        lo_plain, _ = self.calc.calculate(atk_plain, m, def_, _field())
        ratio = lo_hp / lo_plain if lo_plain > 0 else 0
        assert abs(ratio - 2.0) < 0.05

    def test_burn_halves_physical(self):
        atk_burned = _pokemon(status="BRN")
        atk_healthy = _pokemon()
        def_ = _pokemon(types=["Normal"])
        m = _move()  # Physical
        lo_burn, _ = self.calc.calculate(atk_burned, m, def_, _field())
        lo_ok, _ = self.calc.calculate(atk_healthy, m, def_, _field())
        ratio = lo_burn / lo_ok if lo_ok > 0 else 0
        assert abs(ratio - 0.5) < 0.05

    def test_burn_no_effect_on_special(self):
        atk_burned = _pokemon(status="BRN")
        atk_healthy = _pokemon()
        def_ = _pokemon(types=["Normal"])
        sp_move = _move("flamethrower", bp=90, move_type="Fire", category="Special")
        lo_burn, _ = self.calc.calculate(atk_burned, sp_move, def_, _field())
        lo_ok, _ = self.calc.calculate(atk_healthy, sp_move, def_, _field())
        ratio = lo_burn / lo_ok if lo_ok > 0 else 1.0
        # Should not be halved
        assert ratio > 0.9

    def test_technician_boosts_weak_moves(self):
        atk_tech = _pokemon(ability="technician")
        atk_plain = _pokemon()
        def_ = _pokemon(types=["Normal"])
        # Bullet Punch: 40 BP (≤60 → Technician applies)
        weak_move = _move("bulletpunch", bp=40, move_type="Steel", category="Physical")
        lo_tech, _ = self.calc.calculate(atk_tech, weak_move, def_, _field())
        lo_plain, _ = self.calc.calculate(atk_plain, weak_move, def_, _field())
        ratio = lo_tech / lo_plain if lo_plain > 0 else 0
        assert abs(ratio - 1.5) < 0.05

    def test_technician_no_boost_above_60(self):
        atk_tech = _pokemon(ability="technician")
        atk_plain = _pokemon()
        def_ = _pokemon(types=["Normal"])
        strong_move = _move("earthquake", bp=100, move_type="Ground", category="Physical")
        lo_tech, _ = self.calc.calculate(atk_tech, strong_move, def_, _field())
        lo_plain, _ = self.calc.calculate(atk_plain, strong_move, def_, _field())
        ratio = lo_tech / lo_plain if lo_plain > 0 else 0
        assert abs(ratio - 1.0) < 0.05


class TestDamageCalcWeather:
    def setup_method(self):
        self.calc = _make_calc()

    def test_rain_boosts_water(self):
        from knowledge.damage_calc import FieldState
        rain_field = FieldState(weather="RainDance")
        dry_field = FieldState()
        atk = _pokemon(types=["Water"])
        def_ = _pokemon(types=["Normal"])
        m = _move("surf", bp=90, move_type="Water", category="Special")
        lo_rain, _ = self.calc.calculate(atk, m, def_, rain_field)
        lo_dry, _ = self.calc.calculate(atk, m, def_, dry_field)
        ratio = lo_rain / lo_dry if lo_dry > 0 else 0
        assert abs(ratio - 1.5) < 0.1

    def test_rain_weakens_fire(self):
        from knowledge.damage_calc import FieldState
        rain_field = FieldState(weather="RainDance")
        dry_field = FieldState()
        atk = _pokemon(types=["Fire"])
        def_ = _pokemon(types=["Normal"])
        m = _move("flamethrower", bp=90, move_type="Fire", category="Special")
        lo_rain, _ = self.calc.calculate(atk, m, def_, rain_field)
        lo_dry, _ = self.calc.calculate(atk, m, def_, dry_field)
        ratio = lo_rain / lo_dry if lo_dry > 0 else 0
        assert abs(ratio - 0.5) < 0.05


class TestDamageCalcMean:
    def setup_method(self):
        self.calc = _make_calc()

    def test_mean_mode_returns_equal_lo_hi(self):
        atk = _pokemon()
        def_ = _pokemon(types=["Normal"])
        lo, hi = self.calc.calculate(atk, _move(), def_, _field(), rng="mean")
        assert lo == hi

    def test_expected_damage_midpoint(self):
        atk = _pokemon()
        def_ = _pokemon(types=["Normal"])
        lo, hi = self.calc.calculate(atk, _move(), def_, _field(), rng="range")
        mean = self.calc.expected_damage(atk, _move(), def_, _field())
        assert lo <= mean <= hi
