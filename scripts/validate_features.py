#!/usr/bin/env python3
"""
D5-ext: Full 959-dimension feature vector validation.

For every .npz trajectory file, validates:
  1. No NaN / Inf (sanity gate)
  2. Per-feature range checks — mismatch taxonomy by feature index
  3. Opponent-slot leakage check — unrevealed slots must be zero
  4. Own-slot ordering stability — species index at each slot must not change
     across turns (except when slot becomes zero after faint)
  5. Legal mask sanity — at least 1 legal action per non-force-switch turn

Feature layout (POKEMON_FEATURES=77, TEAM_SIZE=6, GLOBAL_FEATURES=35):
  Slots 0-5:  own team  (offset = slot * 77)
  Slots 6-11: opp team  (offset = slot * 77)
  [924:]      global    (35 features)

Per-slot feature indices (relative to slot start):
  [0]      species_idx         int, [0, 508]
  [1]      hp_fraction         float, [0, 1]
  [2]      is_active           binary {0,1}
  [3-9]    status one-hot      binary, sum<=1
  [10-16]  boosts              [-1, 1]
  [17-22]  base stats /255     [0, 1]
  [23]     spe /200            [0, 1.5]  (spe>200 is allowed but rare)
  [24-41]  move types          binary {0,1}
  [42-46]  move flags          binary {0,1}
  [47-54]  item one-hot        binary, sum<=1
  [55]     best_dmg_norm       [0, 2]   (arbitrary cap; 1 = OHKO)
  [56]     can_tera            binary {0,1}
  [57-74]  tera type           binary {0,1}
  [75]     times_active /10    [0, 1]
  [76]     revealed            binary {0,1}

Usage:
  cd /home/user/showdown-bot
  python3 scripts/validate_features.py [--max 200]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

POKEMON_FEATURES = 77
TEAM_SIZE = 6
GLOBAL_FEATURES = 35
FEATURE_DIM = TEAM_SIZE * 2 * POKEMON_FEATURES + GLOBAL_FEATURES  # 959

OWN_START = 0
OPP_START = TEAM_SIZE * POKEMON_FEATURES   # 462
GLOBAL_START = TEAM_SIZE * 2 * POKEMON_FEATURES  # 924


def _slot_offset(slot: int) -> int:
    return slot * POKEMON_FEATURES


def _slot_range(slot: int) -> tuple[int, int]:
    s = _slot_offset(slot)
    return s, s + POKEMON_FEATURES


# ---------------------------------------------------------------------------
# Per-feature check definitions
# Returns (check_name, bad_mask) for each feature in the slot vector
# ---------------------------------------------------------------------------

def _check_slot(sv: np.ndarray, is_opp: bool) -> dict[int, list[str]]:
    """
    Validate a (T, 77) array of slot features.  Returns {rel_idx: [error_strs]}.

    Own slots (is_opp=False): move-type/flag/item/tera features are binary {0,1}.
    Opp slots (is_opp=True) : those same features are *continuous* probabilities
    from SetPredictor — valid range is [0, 1], not necessarily {0, 1}.
    """
    errs: dict[int, list[str]] = {}

    def flag(rel_idx, reason):
        errs.setdefault(rel_idx, []).append(reason)

    # [0] species_idx: int in [0, 508]; zero only if slot is empty (all zeros)
    s0 = sv[:, 0]
    nonempty = sv.sum(axis=1) != 0
    bad = nonempty & ((s0 < 0) | (s0 > 508))
    if bad.any():
        flag(0, f"{bad.sum()} turns: species_idx out of [0,508]; got min={s0[nonempty].min():.0f} max={s0[nonempty].max():.0f}")

    # [1] hp_fraction: [0, 1]
    hp = sv[:, 1]
    bad = (hp < 0) | (hp > 1.001)
    if bad.any():
        flag(1, f"{bad.sum()} turns: hp_fraction out of [0,1]; got {hp[bad][:5]}")

    # [2] is_active: binary
    ia = sv[:, 2]
    bad = (ia != 0) & (ia != 1)
    if bad.any():
        flag(2, f"{bad.sum()} turns: is_active not binary")

    # [3-9] status one-hot: each binary, sum<=1
    status = sv[:, 3:10]
    bad_vals = ((status != 0) & (status != 1)).any(axis=1)
    if bad_vals.any():
        flag(3, f"{bad_vals.sum()} turns: status not binary")
    bad_sum = status.sum(axis=1) > 1
    if bad_sum.any():
        flag(3, f"{bad_sum.sum()} turns: status one-hot sum > 1")

    # [10-16] boosts: [-1, 1]
    boosts = sv[:, 10:17]
    bad = (boosts < -1.001) | (boosts > 1.001)
    if bad.any():
        flag(10, f"{bad.any(axis=1).sum()} turns: boost out of [-1,1]")

    # [17-22] base stats /255: [0, 1]
    bstats = sv[:, 17:23]
    bad = (bstats < 0) | (bstats > 1.001)
    if bad.any():
        flag(17, f"{bad.any(axis=1).sum()} turns: base_stat/255 out of [0,1]")

    # [23] spe/200: [0, 2] — opponent spe can be scaled by Scarf prob so cap is 2.0
    spe = sv[:, 23]
    spe_cap = 2.01 if is_opp else 1.51
    bad = (spe < 0) | (spe > spe_cap)
    if bad.any():
        flag(23, f"{bad.sum()} turns: spe/200 out of [0,{spe_cap:.1f}]; got max={spe.max():.3f}")

    # [24-41] move types: binary for own, continuous [0,1] for opp (SetPredictor probs)
    mtypes = sv[:, 24:42]
    if is_opp:
        bad = (mtypes < 0) | (mtypes > 1.001)
        if bad.any():
            flag(24, f"{bad.any(axis=1).sum()} turns: opp move_type prob out of [0,1]")
    else:
        bad = (mtypes != 0) & (mtypes != 1)
        if bad.any():
            flag(24, f"{bad.any(axis=1).sum()} turns: move_type not binary")

    # [42-46] move flags: binary for own, continuous [0,1] for opp
    mflags = sv[:, 42:47]
    if is_opp:
        bad = (mflags < 0) | (mflags > 1.001)
        if bad.any():
            flag(42, f"{bad.any(axis=1).sum()} turns: opp move_flag prob out of [0,1]")
    else:
        bad = (mflags != 0) & (mflags != 1)
        if bad.any():
            flag(42, f"{bad.any(axis=1).sum()} turns: move_flag not binary")

    # [47-54] item: binary+sum<=1 for own; continuous [0,1] for opp
    items = sv[:, 47:55]
    if is_opp:
        bad = (items < 0) | (items > 1.001)
        if bad.any():
            flag(47, f"{bad.any(axis=1).sum()} turns: opp item prob out of [0,1]")
    else:
        bad_vals = ((items != 0) & (items != 1)).any(axis=1)
        if bad_vals.any():
            flag(47, f"{bad_vals.sum()} turns: item not binary")
        bad_sum = items.sum(axis=1) > 1
        if bad_sum.any():
            flag(47, f"{bad_sum.sum()} turns: item one-hot sum > 1")

    # [55] best_dmg_norm: [0, 2]
    dmg = sv[:, 55]
    bad = (dmg < 0) | (dmg > 2.001)
    if bad.any():
        flag(55, f"{bad.sum()} turns: best_dmg out of [0,2]; got max={dmg.max():.3f}")

    # [56] can_tera: binary
    ct = sv[:, 56]
    bad = (ct != 0) & (ct != 1)
    if bad.any():
        flag(56, f"{bad.sum()} turns: can_tera not binary")

    # [57-74] tera type: binary for own; continuous [0,1] for opp (SetPredictor)
    tera = sv[:, 57:75]
    if is_opp:
        bad = (tera < 0) | (tera > 1.001)
        if bad.any():
            flag(57, f"{bad.any(axis=1).sum()} turns: opp tera_type prob out of [0,1]")
    else:
        bad = (tera != 0) & (tera != 1)
        if bad.any():
            flag(57, f"{bad.any(axis=1).sum()} turns: tera_type not binary")

    # [75] times_active: [0, 1]
    ta = sv[:, 75]
    bad = (ta < 0) | (ta > 1.001)
    if bad.any():
        flag(75, f"{bad.sum()} turns: times_active/10 out of [0,1]")

    # [76] revealed: binary
    rev = sv[:, 76]
    bad = (rev != 0) & (rev != 1)
    if bad.any():
        flag(76, f"{bad.sum()} turns: revealed not binary")

    # Opponent-slot leakage: unrevealed slot must be entirely zero
    if is_opp:
        unrevealed = sv[:, 76] == 0
        # species_idx must be 0 for unrevealed slots
        leaking = unrevealed & (sv[:, 0] > 0)
        if leaking.any():
            flag(0, f"LEAKAGE: {leaking.sum()} turns: opp slot unrevealed but species_idx>0")
        # hp must be 0 for unrevealed slots
        leaking_hp = unrevealed & (sv[:, 1] > 0)
        if leaking_hp.any():
            flag(1, f"LEAKAGE: {leaking_hp.sum()} turns: opp slot unrevealed but hp>0")

    return errs


def _check_global(gv: np.ndarray) -> dict[int, list[str]]:
    """Validate (T, 35) global feature array."""
    errs: dict[int, list[str]] = {}

    def flag(rel_idx, reason):
        errs.setdefault(rel_idx, []).append(reason)

    # [0-4] weather one-hot: sum==1
    weather = gv[:, 0:5]
    bad = (weather != 0) & (weather != 1)
    if bad.any():
        flag(0, f"{bad.any(axis=1).sum()} turns: weather not binary")
    bad_sum = weather.sum(axis=1) != 1
    if bad_sum.any():
        flag(0, f"{bad_sum.sum()} turns: weather one-hot sum != 1")

    # [5-9] terrain one-hot: sum==1
    terrain = gv[:, 5:10]
    bad = (terrain != 0) & (terrain != 1)
    if bad.any():
        flag(5, f"{bad.any(axis=1).sum()} turns: terrain not binary")
    bad_sum = terrain.sum(axis=1) != 1
    if bad_sum.any():
        flag(5, f"{bad_sum.sum()} turns: terrain one-hot sum != 1")

    # [10-11] trick room: binary + [0,1]
    tr = gv[:, 10]
    bad = (tr != 0) & (tr != 1)
    if bad.any():
        flag(10, f"{bad.sum()} turns: trick_room not binary")
    tr_rem = gv[:, 11]
    bad = (tr_rem < 0) | (tr_rem > 1.001)
    if bad.any():
        flag(11, f"{bad.sum()} turns: trick_room_remaining out of [0,1]")

    # [12-19] hazards: [0, 1]
    haz = gv[:, 12:20]
    bad = (haz < 0) | (haz > 1.001)
    if bad.any():
        flag(12, f"{bad.any(axis=1).sum()} turns: hazard feature out of [0,1]")

    # [20-31] screens: [0, 1]
    scr = gv[:, 20:32]
    bad = (scr < 0) | (scr > 1.001)
    if bad.any():
        flag(20, f"{bad.any(axis=1).sum()} turns: screen feature out of [0,1]")

    # [32] turn/100: [0, 1] (clipped in extractor; may exceed if >100 turns)
    tval = gv[:, 32]
    bad = (tval < 0) | (tval > 1.001)
    if bad.any():
        flag(32, f"{bad.sum()} turns: turn/100 out of [0,1]")

    # [33-34] remaining /6: [0, 1]
    rem = gv[:, 33:35]
    bad = (rem < 0) | (rem > 1.001)
    if bad.any():
        flag(33, f"{bad.any(axis=1).sum()} turns: remaining/6 out of [0,1]")

    return errs


def validate_file(npz_path: Path) -> dict:
    result = {
        "path": str(npz_path),
        "ok": False,
        "turns": 0,
        "nan_inf": False,
        "range_errors": {},  # abs_feature_idx -> [error_strings]
        "stability_errors": 0,
        "legal_mask_errors": 0,
    }

    try:
        d = np.load(npz_path)
        states = d["states"]   # (T, 959)
        actions = d["actions"]  # (T,)
        masks = d["legal_masks"]  # (T, 13)
        force_sw = d["force_switch"]  # (T,) bool

        result["turns"] = len(states)

        # 1. NaN / Inf gate
        if np.any(np.isnan(states)) or np.any(np.isinf(states)):
            result["nan_inf"] = True
            return result

        # 2. Per-slot range checks
        for slot in range(TEAM_SIZE * 2):
            s_abs, e_abs = _slot_range(slot)
            sv = states[:, s_abs:e_abs]
            is_opp = slot >= TEAM_SIZE
            slot_errs = _check_slot(sv, is_opp)
            for rel_idx, msgs in slot_errs.items():
                abs_idx = s_abs + rel_idx
                result["range_errors"].setdefault(abs_idx, []).extend(msgs)

        # 3. Global feature range checks
        gv = states[:, GLOBAL_START:]
        global_errs = _check_global(gv)
        for rel_idx, msgs in global_errs.items():
            abs_idx = GLOBAL_START + rel_idx
            result["range_errors"].setdefault(abs_idx, []).extend(msgs)

        # 4. Own-slot ordering stability: own species_idx must not change across
        #    turns (once non-zero) — fainted slots can return to zero afterward.
        for slot in range(TEAM_SIZE):
            abs_idx = _slot_offset(slot)
            species_col = states[:, abs_idx]
            non_zero = species_col[species_col != 0]
            if len(non_zero) > 0:
                first_species = non_zero[0]
                bad_changes = (
                    (species_col != 0) & (species_col != first_species)
                ).sum()
                if bad_changes > 0:
                    result["stability_errors"] += bad_changes

        # 5. Legal mask sanity: every non-force-switch turn must have ≥1 legal action
        non_forced = ~force_sw
        legal_count = masks[non_forced].sum(axis=1)
        bad_legal = (legal_count == 0).sum()
        result["legal_mask_errors"] = int(bad_legal)

        result["ok"] = True
    except Exception as e:
        result["range_errors"][-1] = [f"exception: {e}"]

    return result


def main():
    parser = argparse.ArgumentParser(description="D5-ext full-vector validator")
    parser.add_argument("--max", type=int, default=0, help="Max .npz files to check (0=all)")
    parser.add_argument(
        "--traj-dir", default="data/trajectories/gen9randombattle",
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    traj_dir = base / args.traj_dir
    npz_files = sorted(traj_dir.glob("*.npz"))
    if args.max > 0:
        npz_files = npz_files[:args.max]

    if not npz_files:
        print(f"No .npz files found in {traj_dir}")
        return 1

    print(f"D5-ext Feature Vector Validator")
    print(f"  Checking {len(npz_files)} trajectory files from {traj_dir}")
    print()

    all_results = []
    for npz_path in npz_files:
        r = validate_file(npz_path)
        all_results.append(r)

    ok_results = [r for r in all_results if r["ok"]]
    total_turns = sum(r["turns"] for r in ok_results)
    total_nan = sum(1 for r in all_results if r["nan_inf"])
    total_stability = sum(r["stability_errors"] for r in ok_results)
    total_legal_err = sum(r["legal_mask_errors"] for r in ok_results)

    print(f"{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Files checked         : {len(all_results)}")
    print(f"  Parse OK              : {len(ok_results)}")
    print(f"  Total turns           : {total_turns:,}")
    print(f"  NaN/Inf files         : {total_nan}")
    print(f"  Slot-order instability: {total_stability} turn×slot violations")
    print(f"  Legal-mask zeros      : {total_legal_err} non-forced turns with no legal action")
    print()

    # Aggregate range errors across all files
    agg_errs: dict[int, int] = {}
    for r in ok_results:
        for abs_idx, msgs in r["range_errors"].items():
            agg_errs[abs_idx] = agg_errs.get(abs_idx, 0) + len(msgs)

    if agg_errs:
        print(f"  Range errors by feature index ({len(agg_errs)} indices affected):")
        for abs_idx in sorted(agg_errs.keys()):
            count = agg_errs[abs_idx]
            # Classify: slot vs global
            if abs_idx < GLOBAL_START:
                slot = abs_idx // POKEMON_FEATURES
                rel = abs_idx % POKEMON_FEATURES
                loc = f"slot {slot} rel[{rel}]"
            else:
                rel = abs_idx - GLOBAL_START
                loc = f"global[{rel}]"
            print(f"    feat[{abs_idx:4d}] ({loc:18s}): {count} errors")
    else:
        print(f"  Range errors: NONE")
    print()

    all_pass = (
        total_nan == 0
        and total_stability == 0
        and total_legal_err == 0
        and len(agg_errs) == 0
        and len(ok_results) == len(all_results)
    )

    print(f"RESULT: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        if total_nan > 0:
            print(f"  ✗ {total_nan} files have NaN/Inf")
        if len(ok_results) < len(all_results):
            print(f"  ✗ {len(all_results) - len(ok_results)} files failed to load")
        if total_stability > 0:
            print(f"  ✗ {total_stability} slot-order stability violations")
        if total_legal_err > 0:
            print(f"  ✗ {total_legal_err} non-forced turns have empty legal mask")
        if agg_errs:
            print(f"  ✗ {sum(agg_errs.values())} range violations across {len(agg_errs)} feature indices")
    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
