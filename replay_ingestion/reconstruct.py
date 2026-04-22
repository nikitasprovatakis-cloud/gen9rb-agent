"""
D3: First-person reconstruction of Showdown replay states.

For each turn in a ParsedBattle, produces two ReconstructedView objects —
one from each player's perspective.  Opponent information is masked to only
what was visible at that point in the replay:

  - Species / HP / status / boosts: shown only after the slot is revealed
    (i.e., the Pokemon has been switched in at least once)
  - moves_used: only moves the opponent has actually played so far
  - item_revealed / ability_revealed / tera_type_revealed: as tracked by parser
    (the parser only sets these when an event publicly exposes them)

Illusion: after the Illusion break, the parser sets slot.forme = true species
and slot.illusion_entry_species = disguised species.  _resolve_slot() promotes
forme → species so downstream consumers always read the current species from
slot.species.  Pre-break snapshots still show the disguised species because
slot.forme is None at that point.

Ditto / Transform: after Transform, slot.forme = target species.  Same
promotion applies — no extra masking needed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from replay_ingestion.parser import (
    ActionRecord,
    FieldState,
    ParsedBattle,
    PokemonSlot,
    SideConditions,
    TurnSnapshot,
)


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class ReconstructedView:
    """
    First-person game state from one player's perspective at one turn.

    own_slots  — full visibility (all HP, status, boosts, moves, etc.)
    opp_slots  — revealed info only; non-revealed slots are empty placeholders
    action     — the choice this player made during this turn (for training labels)
    """
    replay_id: str
    turn_number: int
    player: int             # 1 or 2

    own_slots: list[PokemonSlot]
    opp_slots: list[PokemonSlot]

    own_active_nick: str
    opp_active_nick: str

    field_state: FieldState
    own_side: SideConditions
    opp_side: SideConditions

    own_can_tera: bool
    is_force_switch: bool
    own_cant: bool    # player was prevented from acting this turn (|cant| event)

    action: Optional[ActionRecord]

    own_team_size: int
    opp_team_size: int

    winner: Optional[int]   # 1, 2, or None


@dataclass
class ReconstructionResult:
    """All first-person views extracted from a single replay."""
    replay_id: str
    p1_views: list[ReconstructedView]   # one per turn, p1 POV
    p2_views: list[ReconstructedView]   # one per turn, p2 POV
    winner: Optional[int]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconstruct(battle: ParsedBattle) -> Optional[ReconstructionResult]:
    """
    Convert a ParsedBattle into per-player, per-turn ReconstructedViews.
    Returns None if the battle has no parseable turns.
    """
    if not battle.turns:
        return None

    p1_views: list[ReconstructedView] = []
    p2_views: list[ReconstructedView] = []

    for snap in battle.turns:
        p1_views.append(_build_view(battle, snap, player=1))
        p2_views.append(_build_view(battle, snap, player=2))

    return ReconstructionResult(
        replay_id=battle.replay_id,
        p1_views=p1_views,
        p2_views=p2_views,
        winner=battle.winner,
    )


def reconstruct_file(json_path: str, replay_id: str = "") -> Optional[ReconstructionResult]:
    """Load a .json replay file, parse it, and reconstruct first-person views."""
    from replay_ingestion.parser import parse_replay_file
    battle = parse_replay_file(json_path, replay_id=replay_id)
    if battle is None:
        return None
    return reconstruct(battle)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_view(
    battle: ParsedBattle,
    snap: TurnSnapshot,
    player: int,
) -> ReconstructedView:
    if player == 1:
        own_raw         = snap.p1_slots
        opp_raw         = snap.p2_slots
        own_active_nick = snap.p1_active_nick
        opp_active_nick = snap.p2_active_nick
        own_side        = snap.p1_side
        opp_side        = snap.p2_side
        own_can_tera    = snap.p1_can_tera
        is_force_switch = snap.p1_force_switch
        own_cant        = snap.p1_cant
        action          = snap.p1_action
        own_team_size   = snap.p1_team_size
        opp_team_size   = snap.p2_team_size
    else:
        own_raw         = snap.p2_slots
        opp_raw         = snap.p1_slots
        own_active_nick = snap.p2_active_nick
        opp_active_nick = snap.p1_active_nick
        own_side        = snap.p2_side
        opp_side        = snap.p1_side
        own_can_tera    = snap.p2_can_tera
        is_force_switch = snap.p2_force_switch
        own_cant        = snap.p2_cant
        action          = snap.p2_action
        own_team_size   = snap.p2_team_size
        opp_team_size   = snap.p1_team_size

    own_slots = [_resolve_slot(s, masked=False) for s in own_raw]
    opp_slots = [_resolve_slot(s, masked=True)  for s in opp_raw]

    return ReconstructedView(
        replay_id=battle.replay_id,
        turn_number=snap.turn_number,
        player=player,
        own_slots=own_slots,
        opp_slots=opp_slots,
        own_active_nick=own_active_nick,
        opp_active_nick=opp_active_nick,
        field_state=copy.deepcopy(snap.field_state),
        own_side=own_side.clone(),
        opp_side=opp_side.clone(),
        own_can_tera=own_can_tera,
        is_force_switch=is_force_switch,
        own_cant=own_cant,
        action=action,
        own_team_size=own_team_size,
        opp_team_size=opp_team_size,
        winner=battle.winner,
    )


def _resolve_slot(slot: PokemonSlot, masked: bool) -> PokemonSlot:
    """
    Return a copy of slot suitable for one player's view.

    masked=True  (opponent): non-revealed slots → placeholder; revealed slots
                 retain only what the parser recorded as publicly visible.
    masked=False (own side): full copy, but forme resolved into species so
                 downstream code always reads current species from slot.species.

    Species resolution:
      slot.forme is set when the species changes mid-battle (forme changes,
      Illusion reveal, or Transform).  We promote forme → species and clear
      the field so consumers only need to read slot.species.
    """
    if masked and not slot.revealed:
        # Pokemon not yet switched in — completely unknown to the opponent
        return PokemonSlot(
            species="",
            nickname="",
            revealed=False,
            is_active=False,
            fainted=False,
        )

    s = copy.deepcopy(slot)

    # Promote current forme → species (handles Illusion, formechange, Transform)
    if s.forme is not None:
        s.species = s.forme
        s.forme = None
    s.illusion_entry_species = None  # internal bookkeeping, not needed downstream

    # For masked (opponent) slots the parser has already ensured that only
    # publicly visible fields are populated:
    #   moves_used         ← only moves actually used in battle
    #   item_revealed      ← only items exposed by activation events
    #   ability_revealed   ← only abilities triggered in battle
    #   tera_type_revealed ← only if opponent terastallized
    #   boosts             ← visible to both sides when they change
    # Nothing further to strip.

    return s
