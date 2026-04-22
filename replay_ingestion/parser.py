"""
D2: Gen 9 Random Battle replay log parser.

Converts a raw Showdown pipe-delimited log into a ParsedBattle containing
a sequence of TurnSnapshot objects — one per turn — capturing the full
game state visible in the log (both sides).  First-person reconstruction
(opponent masking) is handled by reconstruct.py.

HP format note: Showdown's replay logs store exact HP values even when
the "HP Percentage Mod" rule is active.  All HP is parsed as current/max
integers; hp_fraction = current / max.

Illusion note: When Illusion breaks (|-end|...|Illusion), the parser
records an IllusionBreak event and retroactively relabels the Pokemon slot
from its disguised species to its true species (Zoroark or Zorua).

Unknown events: Silently counted via `unknown_event_counts`; never raises.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ActionRecord:
    """Records one player's primary decision for a turn."""
    action_type: str        # "move" | "switch" | "drag"
    name: str               # move name (e.g. "Earthquake") or switch target species
    nickname: str           # nickname of acting / incoming Pokemon
    player: int             # 1 or 2
    is_tera: bool = False   # player Terastallized this move
    is_forced: bool = False # drag or force-switch after faint


@dataclass
class PokemonSlot:
    """Live state of one Pokemon in battle (from log perspective)."""
    species: str              # base species name, e.g. "Serperior"
    nickname: str             # nickname as it appears in the log
    level: int = 100
    gender: Optional[str] = None  # "M" | "F" | None

    max_hp: int = 100
    current_hp: int = 100

    status: Optional[str] = None  # None | "par" | "brn" | "slp" | "frz" | "tox" | "psn" | "fnt"
    boosts: dict = field(default_factory=lambda: {
        "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0,
        "accuracy": 0, "evasion": 0,
    })
    is_active: bool = False
    fainted: bool = False
    revealed: bool = False   # has appeared in battle

    # What has been revealed about this Pokemon so far
    moves_used: list[str] = field(default_factory=list)    # move names, in order used
    item_revealed: Optional[str] = None
    ability_revealed: Optional[str] = None
    tera_type_revealed: Optional[str] = None  # e.g. "Fire"
    is_terastallized: bool = False
    forme: Optional[str] = None  # current forme if different from entry species

    # Illusion tracking: did this slot *enter* disguised as another species?
    illusion_entry_species: Optional[str] = None   # what it appeared as before break

    @property
    def hp_fraction(self) -> float:
        if self.fainted:
            return 0.0
        if self.max_hp == 0:
            return 1.0
        return max(0.0, min(1.0, self.current_hp / self.max_hp))

    def clone(self) -> "PokemonSlot":
        """Deep-copy for snapshotting."""
        import copy
        return copy.deepcopy(self)


@dataclass
class FieldState:
    """Global field conditions."""
    weather: Optional[str] = None     # "RainDance" | "SunnyDay" | "Sandstorm" | "Snow" | None
    weather_turn: int = 0             # turn weather was set

    terrain: Optional[str] = None    # "Electric" | "Grassy" | "Misty" | "Psychic" | None
    terrain_turn: int = 0

    trick_room: bool = False
    trick_room_turn: int = 0


@dataclass
class SideConditions:
    """Side-specific conditions."""
    stealth_rock: bool = False
    spikes: int = 0          # layers: 0–3
    toxic_spikes: int = 0    # layers: 0–2
    sticky_web: bool = False
    reflect: int = 0         # turns remaining (5 or 8 with Light Clay), stored as activation turn
    light_screen: int = 0
    aurora_veil: int = 0
    # Store as activation turn (0 = not active); reconstruct.py computes turns remaining
    reflect_turn: int = 0
    light_screen_turn: int = 0
    aurora_veil_turn: int = 0

    def clone(self) -> "SideConditions":
        import copy
        return copy.deepcopy(self)


@dataclass
class TurnSnapshot:
    """Full game state at the START of a turn (after all prior-turn effects settle)."""
    turn_number: int

    p1_slots: list[PokemonSlot] = field(default_factory=list)  # ordered by reveal
    p2_slots: list[PokemonSlot] = field(default_factory=list)

    p1_active_nick: str = ""  # nickname of active Pokemon
    p2_active_nick: str = ""

    field_state: FieldState = field(default_factory=FieldState)
    p1_side: SideConditions = field(default_factory=SideConditions)
    p2_side: SideConditions = field(default_factory=SideConditions)

    p1_can_tera: bool = True   # False once used
    p2_can_tera: bool = True

    p1_force_switch: bool = False  # this turn is a force-switch for p1
    p2_force_switch: bool = False

    # Action taken during this turn (populated during event processing)
    p1_action: Optional[ActionRecord] = None
    p2_action: Optional[ActionRecord] = None

    # Metadata
    p1_team_size: int = 6
    p2_team_size: int = 6


@dataclass
class ParsedBattle:
    """Fully parsed replay ready for reconstruction."""
    replay_id: str
    format: str = "gen9randombattle"
    gen: int = 9

    p1_username: str = ""
    p2_username: str = ""
    p1_rating: Optional[int] = None
    p2_rating: Optional[int] = None

    # One snapshot per turn (index = turn_number - 1)
    turns: list[TurnSnapshot] = field(default_factory=list)

    winner: Optional[int] = None  # 1, 2, or None (tie / unfinished)
    upload_time: int = 0

    unknown_event_counts: dict = field(default_factory=dict)
    parse_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser internals
# ---------------------------------------------------------------------------

_STATUS_MAP = {
    "par": "par", "brn": "brn", "slp": "slp",
    "frz": "frz", "tox": "tox", "psn": "psn", "fnt": "fnt",
}

_STAT_MAP = {
    "atk": "atk", "def": "def", "spa": "spa",
    "spd": "spd", "spe": "spe",
    "accuracy": "accuracy", "evasion": "evasion",
    # Showdown sometimes spells these out
    "attack": "atk", "defense": "def", "specialattack": "spa",
    "specialdefense": "spd", "speed": "spe",
}

_WEATHER_MAP = {
    "RainDance": "RainDance",
    "SunnyDay": "SunnyDay",
    "Sandstorm": "Sandstorm",
    "Snow": "Snow",
    "Snowscape": "Snow",     # alias
    "PrimordialSea": "RainDance",
    "DesolateLand": "SunnyDay",
    "DeltaStream": None,     # neutral weather
    "none": None,
    "": None,
}

# Events that carry no game-state information for our purposes
_IGNORE = frozenset({
    "", "t:", "j", "J", "l", "L", "n", "c", "c:", "chat", "chatmsg",
    "chatmsg-raw", "raw", "html", "uhtml", "uhtmlchange",
    "join", "leave",
    "inactive", "inactiveoff", "timer",
    "gametype", "gen", "tier", "rated", "rule", "seed", "teampreview",
    "start", "upkeep",
    "badge", "bigerror", "debug", "deinit", "error", "init",
    "askreg", "hidelines", "unlink",
    "request", "sentchoice",
    "-anim", "-hint", "-message", "-notarget", "-nothing",
    "-ohko", "-prepare", "-primal", "-zbroken",
    "-singlemove", "-singleturn",       # no lasting state change needed
    "-crit", "-supereffective", "-resisted", "-immune",
    "-hitcount", "-miss", "-fail",
    "-center", "-fieldactivate",
    "message",
})


class Gen9Parser:
    """
    Parses a Gen 9 Random Battle replay log into a ParsedBattle.

    Usage:
        parser = Gen9Parser()
        battle = parser.parse(log_text, replay_id="gen9randombattle-123456")
        if battle is None:
            print("Parse failed")
    """

    # ── public entry point ────────────────────────────────────────────────────

    def parse(self, log_text: str, replay_id: str = "", upload_time: int = 0) -> Optional[ParsedBattle]:
        battle = ParsedBattle(replay_id=replay_id, upload_time=upload_time)

        # Live state (modified as we process events)
        p1_slots: list[PokemonSlot] = []
        p2_slots: list[PokemonSlot] = []
        field = FieldState()
        p1_side = SideConditions()
        p2_side = SideConditions()
        p1_can_tera = True
        p2_can_tera = True
        p1_active_nick = ""
        p2_active_nick = ""
        p1_team_size = 6
        p2_team_size = 6

        current_turn = 0           # 0 = pre-battle
        current_snap: Optional[TurnSnapshot] = None  # snapshot being built
        # Tera flag: set when |-terastallize| is seen, consumed by next |move|
        pending_tera_p1 = False
        pending_tera_p2 = False
        # Force-switch flag: set after a faint or pivot switch-out
        force_switch_p1 = False
        force_switch_p2 = False

        lines = log_text.splitlines()

        for raw_line in lines:
            raw_line = raw_line.strip()
            if not raw_line or raw_line == "|":
                continue

            parts = raw_line.split("|")
            if not parts or parts[0] != "":
                continue   # not a protocol line
            parts = parts[1:]  # drop leading empty

            if not parts:
                continue
            event = parts[0]
            args = parts[1:]

            # ── turn boundary ────────────────────────────────────────────────
            if event == "turn":
                # Finalize previous snapshot
                if current_snap is not None:
                    battle.turns.append(current_snap)

                current_turn = int(args[0]) if args else current_turn + 1

                # Snapshot current state for this new turn
                current_snap = TurnSnapshot(
                    turn_number=current_turn,
                    p1_slots=[s.clone() for s in p1_slots],
                    p2_slots=[s.clone() for s in p2_slots],
                    p1_active_nick=p1_active_nick,
                    p2_active_nick=p2_active_nick,
                    field_state=_clone(field),
                    p1_side=p1_side.clone(),
                    p2_side=p2_side.clone(),
                    p1_can_tera=p1_can_tera,
                    p2_can_tera=p2_can_tera,
                    p1_force_switch=force_switch_p1,
                    p2_force_switch=force_switch_p2,
                    p1_team_size=p1_team_size,
                    p2_team_size=p2_team_size,
                )
                # Reset turn-specific state
                force_switch_p1 = False
                force_switch_p2 = False
                continue

            # ── header events ─────────────────────────────────────────────────
            elif event == "player":
                if len(args) >= 2:
                    slot = args[0]
                    username = args[1]
                    rating_str = args[3] if len(args) > 3 else ""
                    rating = _parse_int_safe(rating_str)
                    if slot == "p1":
                        battle.p1_username = username
                        battle.p1_rating = rating
                    elif slot == "p2":
                        battle.p2_username = username
                        battle.p2_rating = rating

            elif event == "teamsize":
                if len(args) >= 2:
                    if args[0] == "p1":
                        p1_team_size = int(args[1])
                    else:
                        p2_team_size = int(args[1])

            # ── switch / drag ─────────────────────────────────────────────────
            elif event in ("switch", "drag", "replace"):
                if len(args) < 3:
                    continue
                player_nick = args[0]   # e.g. "p1a: Vaporeon"
                details = args[1]       # e.g. "Vaporeon, L86, M"
                hp_str = args[2]        # e.g. "364/364" or "100/100 par"
                player, nick = _parse_player_nick(player_nick)
                species, level, gender = _parse_details(details)
                cur_hp, max_hp, status = _parse_hp(hp_str)
                is_forced = (event == "drag")

                slots = p1_slots if player == 1 else p2_slots
                active_nick = p1_active_nick if player == 1 else p2_active_nick

                # On switch-out: clear boosts, reset volatile effects (tracked via revealed)
                for s in slots:
                    if s.nickname == active_nick and s.is_active:
                        s.is_active = False
                        s.boosts = {k: 0 for k in s.boosts}
                        s.is_terastallized = False  # tera state reset on switch (actually persists, but hp does)
                        break

                # Find existing slot or create new one
                existing = _find_slot(slots, nick)
                if existing is None:
                    slot_obj = PokemonSlot(
                        species=species, nickname=nick,
                        level=level, gender=gender,
                        max_hp=max_hp, current_hp=cur_hp,
                        status=status, revealed=True, is_active=True,
                    )
                    slots.append(slot_obj)
                else:
                    # Update with fresh switch-in data
                    existing.species = species
                    existing.level = level
                    existing.gender = gender
                    existing.max_hp = max_hp
                    existing.current_hp = cur_hp
                    existing.status = status
                    existing.revealed = True
                    existing.is_active = True
                    existing.fainted = False
                    existing.boosts = {k: 0 for k in existing.boosts}
                    slot_obj = existing

                if player == 1:
                    p1_active_nick = nick
                else:
                    p2_active_nick = nick

                # Record action on current snapshot
                if current_snap is not None:
                    action = ActionRecord(
                        action_type="drag" if is_forced else "switch",
                        name=species, nickname=nick,
                        player=player, is_forced=is_forced,
                    )
                    if player == 1:
                        if current_snap.p1_action is None:
                            current_snap.p1_action = action
                    else:
                        if current_snap.p2_action is None:
                            current_snap.p2_action = action

            # ── move ──────────────────────────────────────────────────────────
            elif event == "move":
                if len(args) < 2:
                    continue
                player_nick = args[0]
                move_name = args[1]
                player, nick = _parse_player_nick(player_nick)

                slots = p1_slots if player == 1 else p2_slots
                slot = _find_slot(slots, nick)
                if slot is not None and move_name not in ("Struggle", "Recharge", ""):
                    if move_name not in slot.moves_used:
                        slot.moves_used.append(move_name)

                # Check for tera flag
                is_tera = (pending_tera_p1 if player == 1 else pending_tera_p2)
                if player == 1:
                    pending_tera_p1 = False
                else:
                    pending_tera_p2 = False

                if current_snap is not None:
                    action = ActionRecord(
                        action_type="move", name=move_name, nickname=nick,
                        player=player, is_tera=is_tera,
                    )
                    if player == 1:
                        if current_snap.p1_action is None:
                            current_snap.p1_action = action
                    else:
                        if current_snap.p2_action is None:
                            current_snap.p2_action = action

            # ── cant (trapped / asleep / etc.) ────────────────────────────────
            elif event == "cant":
                # |cant|p1a: Nick|reason|move (optional)
                if not args:
                    continue
                player, nick = _parse_player_nick(args[0])
                # No state change needed beyond tracking "no action"

            # ── damage / heal / sethp ─────────────────────────────────────────
            elif event == "-damage":
                if len(args) < 2:
                    continue
                player, nick = _parse_player_nick(args[0])
                cur_hp, max_hp, status = _parse_hp(args[1])
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None:
                    slot.current_hp = cur_hp
                    if max_hp > 0:
                        slot.max_hp = max_hp
                    if status == "fnt":
                        slot.fainted = True
                        slot.current_hp = 0
                        slot.status = "fnt"
                    elif status is not None:
                        slot.status = status
                # Reveal item from damage source annotation
                _maybe_reveal_item(args, p1_slots, p2_slots)

            elif event == "-heal":
                if len(args) < 2:
                    continue
                player, nick = _parse_player_nick(args[0])
                cur_hp, max_hp, status = _parse_hp(args[1])
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None:
                    slot.current_hp = cur_hp
                    if max_hp > 0:
                        slot.max_hp = max_hp
                    if status is not None:
                        slot.status = status
                _maybe_reveal_item(args, p1_slots, p2_slots)

            elif event == "-sethp":
                if len(args) < 2:
                    continue
                player, nick = _parse_player_nick(args[0])
                cur_hp, max_hp, _ = _parse_hp(args[1])
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None:
                    slot.current_hp = cur_hp
                    if max_hp > 0:
                        slot.max_hp = max_hp

            # ── status apply / cure ───────────────────────────────────────────
            elif event == "-status":
                if len(args) < 2:
                    continue
                player, nick = _parse_player_nick(args[0])
                status_raw = args[1].lower()
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None:
                    slot.status = _STATUS_MAP.get(status_raw, status_raw)
                _maybe_reveal_item(args, p1_slots, p2_slots)

            elif event in ("-curestatus", "-cureteam"):
                if not args:
                    continue
                player, nick = _parse_player_nick(args[0])
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None and slot.status != "fnt":
                    slot.status = None
                _maybe_reveal_item(args, p1_slots, p2_slots)

            # ── stat boosts ───────────────────────────────────────────────────
            elif event == "-boost":
                if len(args) < 3:
                    continue
                player, nick = _parse_player_nick(args[0])
                stat = _STAT_MAP.get(args[1].lower(), args[1].lower())
                amount = _parse_int_safe(args[2])
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None and stat in slot.boosts:
                    slot.boosts[stat] = max(-6, min(6, slot.boosts[stat] + amount))

            elif event == "-unboost":
                if len(args) < 3:
                    continue
                player, nick = _parse_player_nick(args[0])
                stat = _STAT_MAP.get(args[1].lower(), args[1].lower())
                amount = _parse_int_safe(args[2])
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None and stat in slot.boosts:
                    slot.boosts[stat] = max(-6, min(6, slot.boosts[stat] - amount))

            elif event == "-setboost":
                if len(args) < 3:
                    continue
                player, nick = _parse_player_nick(args[0])
                stat = _STAT_MAP.get(args[1].lower(), args[1].lower())
                amount = _parse_int_safe(args[2])
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None and stat in slot.boosts:
                    slot.boosts[stat] = max(-6, min(6, amount))

            elif event in ("-clearboost", "-clearnegativeboost", "-clearpositiveboost"):
                if not args:
                    continue
                player, nick = _parse_player_nick(args[0])
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None:
                    slot.boosts = {k: 0 for k in slot.boosts}

            elif event == "-clearallboost":
                for s in p1_slots + p2_slots:
                    s.boosts = {k: 0 for k in s.boosts}

            elif event == "-invertboost":
                if not args:
                    continue
                player, nick = _parse_player_nick(args[0])
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None:
                    slot.boosts = {k: -v for k, v in slot.boosts.items()}

            elif event == "-copyboost":
                # |-copyboost|target|source — target copies source boosts
                if len(args) < 2:
                    continue
                tplayer, tnick = _parse_player_nick(args[0])
                splayer, snick = _parse_player_nick(args[1])
                tslot = _find_slot(p1_slots if tplayer == 1 else p2_slots, tnick)
                sslot = _find_slot(p1_slots if splayer == 1 else p2_slots, snick)
                if tslot is not None and sslot is not None:
                    tslot.boosts = dict(sslot.boosts)

            elif event == "-swapboost":
                # |-swapboost|p1|p2|stats
                if len(args) < 2:
                    continue
                p1r, n1 = _parse_player_nick(args[0])
                p2r, n2 = _parse_player_nick(args[1])
                s1 = _find_slot(p1_slots if p1r == 1 else p2_slots, n1)
                s2 = _find_slot(p1_slots if p2r == 1 else p2_slots, n2)
                if s1 and s2:
                    s1.boosts, s2.boosts = dict(s2.boosts), dict(s1.boosts)

            # ── weather ───────────────────────────────────────────────────────
            elif event == "-weather":
                weather_raw = args[0] if args else ""
                mapped = _WEATHER_MAP.get(weather_raw)
                field.weather = mapped
                if mapped is not None:
                    field.weather_turn = current_turn
                _maybe_reveal_ability(args, p1_slots, p2_slots)

            # ── terrain / field ───────────────────────────────────────────────
            elif event == "-fieldstart":
                condition = args[0] if args else ""
                if "Terrain" in condition:
                    terrain_name = condition.replace(" Terrain", "").strip()
                    field.terrain = terrain_name
                    field.terrain_turn = current_turn
                elif "Trick Room" in condition or condition == "move: Trick Room":
                    field.trick_room = not field.trick_room  # toggles
                    field.trick_room_turn = current_turn
                # Other field effects (Magic Room, Wonder Room) — ignore for now

            elif event == "-fieldend":
                condition = args[0] if args else ""
                if "Terrain" in condition:
                    field.terrain = None
                elif "Trick Room" in condition or condition == "move: Trick Room":
                    field.trick_room = False
                    field.trick_room_turn = 0

            # ── side conditions ───────────────────────────────────────────────
            elif event == "-sidestart":
                if len(args) < 2:
                    continue
                side_raw = args[0]  # "p1: username" or "p2: username"
                condition = args[1]
                player = 1 if side_raw.startswith("p1") else 2
                side = p1_side if player == 1 else p2_side
                _apply_side_start(side, condition, current_turn)

            elif event == "-sideend":
                if len(args) < 2:
                    continue
                side_raw = args[0]
                condition = args[1]
                player = 1 if side_raw.startswith("p1") else 2
                side = p1_side if player == 1 else p2_side
                _apply_side_end(side, condition)

            elif event == "-swapsideconditions":
                # Court Change
                p1_side, p2_side = p2_side, p1_side

            # ── ability reveal ────────────────────────────────────────────────
            elif event == "-ability":
                if len(args) < 2:
                    continue
                player, nick = _parse_player_nick(args[0])
                ability = args[1]
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None and slot.ability_revealed is None:
                    slot.ability_revealed = ability

            elif event == "-endability":
                pass  # ability lost (Skill Swap etc.), too rare to track

            # ── item reveal ───────────────────────────────────────────────────
            elif event == "-item":
                if len(args) < 2:
                    continue
                player, nick = _parse_player_nick(args[0])
                item = args[1]
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None and slot.item_revealed is None:
                    slot.item_revealed = item

            elif event == "-enditem":
                # Item consumed or destroyed — it was revealed in the process
                if len(args) < 2:
                    continue
                player, nick = _parse_player_nick(args[0])
                item = args[1]
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None:
                    if slot.item_revealed is None:
                        slot.item_revealed = item
                    slot.item_revealed = None   # item is gone after enditem

            # ── forme / detail changes ────────────────────────────────────────
            elif event in ("-formechange", "detailschange", "-terachange"):
                if len(args) < 2:
                    continue
                player, nick = _parse_player_nick(args[0])
                new_species, lv, gd = _parse_details(args[1])
                hp_str = args[2] if len(args) > 2 else ""
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None:
                    slot.forme = new_species
                    if hp_str:
                        cur_hp, max_hp, status = _parse_hp(hp_str)
                        slot.current_hp = cur_hp
                        if max_hp > 0:
                            slot.max_hp = max_hp

            # ── tera ──────────────────────────────────────────────────────────
            elif event == "-terastallize":
                if len(args) < 2:
                    continue
                player, nick = _parse_player_nick(args[0])
                tera_type = args[1]  # e.g. "Fire"
                slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                if slot is not None:
                    slot.tera_type_revealed = tera_type
                    slot.is_terastallized = True
                # Set tera flag for the next |move| event on this player
                if player == 1:
                    pending_tera_p1 = True
                    p1_can_tera = False
                else:
                    pending_tera_p2 = True
                    p2_can_tera = False

            # ── transform ─────────────────────────────────────────────────────
            elif event == "-transform":
                # |-transform|p1a: Ditto|p2a: Garchomp
                # Ditto copies Garchomp's visible data (moves revealed so far, etc.)
                if len(args) < 2:
                    continue
                tp, tnick = _parse_player_nick(args[0])
                sp, snick = _parse_player_nick(args[1])
                tslot = _find_slot(p1_slots if tp == 1 else p2_slots, tnick)
                sslot = _find_slot(p1_slots if sp == 1 else p2_slots, snick)
                if tslot is not None and sslot is not None:
                    # Record which species it transformed into via forme
                    tslot.forme = sslot.species

            # ── Illusion break ────────────────────────────────────────────────
            elif event == "-end":
                # |-end|p1a: Nick|Illusion
                if len(args) >= 2 and args[1] == "Illusion":
                    player, nick = _parse_player_nick(args[0])
                    slots = p1_slots if player == 1 else p2_slots
                    slot = _find_slot(slots, nick)
                    if slot is not None:
                        # The Pokemon was appearing as `nick` (which is the disguised
                        # species' name). The NEXT detailschange or switch event will
                        # reveal the true species. We record what it appeared as.
                        slot.illusion_entry_species = slot.species
                        # The true species is revealed via the next detailschange event
                        # emitted automatically by Showdown after Illusion breaks.
                        logger.debug(
                            "Illusion break detected: %s appeared as %s (player %d)",
                            nick, slot.species, player,
                        )

            # ── faint ─────────────────────────────────────────────────────────
            elif event == "faint":
                if not args:
                    continue
                player, nick = _parse_player_nick(args[0])
                slots = p1_slots if player == 1 else p2_slots
                slot = _find_slot(slots, nick)
                if slot is not None:
                    slot.fainted = True
                    slot.current_hp = 0
                    slot.status = "fnt"
                    slot.is_active = False
                # Next turn will be a force-switch for this player
                if player == 1:
                    force_switch_p1 = True
                else:
                    force_switch_p2 = True

            # ── game end ──────────────────────────────────────────────────────
            elif event == "win":
                winner_name = args[0] if args else ""
                if winner_name == battle.p1_username:
                    battle.winner = 1
                elif winner_name == battle.p2_username:
                    battle.winner = 2
                else:
                    # Try matching by username case-insensitively
                    if winner_name.lower() == battle.p1_username.lower():
                        battle.winner = 1
                    else:
                        battle.winner = 2

            elif event == "tie":
                battle.winner = None  # explicit tie

            # ── ignored events ────────────────────────────────────────────────
            elif event in _IGNORE:
                pass

            # ── unknown events ────────────────────────────────────────────────
            else:
                battle.unknown_event_counts[event] = (
                    battle.unknown_event_counts.get(event, 0) + 1
                )

        # Append the last open snapshot
        if current_snap is not None:
            battle.turns.append(current_snap)

        if not battle.turns:
            return None  # empty / malformed log

        return battle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_player_nick(s: str) -> tuple[int, str]:
    """Parse "p1a: Nickname" → (1, "Nickname")."""
    if ":" not in s:
        return 0, s
    player_part, nick = s.split(":", 1)
    player_num = 1 if "1" in player_part else 2
    return player_num, nick.strip()


def _parse_details(details: str) -> tuple[str, int, Optional[str]]:
    """Parse "Species, L86, M" → ("Species", 86, "M")."""
    parts = [p.strip() for p in details.split(",")]
    if not parts:
        return details.strip(), 100, None
    species = parts[0].strip()
    level = 100
    gender = None
    for p in parts[1:]:
        p = p.strip()
        if p.startswith("L") and p[1:].isdigit():
            level = int(p[1:])
        elif p in ("M", "F"):
            gender = p
        # "shiny", "tera:Type" etc. — ignore
    return species, level, gender


def _parse_hp(hp_str: str) -> tuple[int, int, Optional[str]]:
    """
    Parse "364/364", "282/310 par", "0 fnt", "100/100" etc.
    Returns (current_hp, max_hp, status_str | None).
    """
    s = hp_str.strip()
    parts = s.split()
    hp_part = parts[0] if parts else "0"
    status_raw = parts[1].lower() if len(parts) > 1 else None
    status = _STATUS_MAP.get(status_raw, None) if status_raw else None

    if hp_part == "0" or hp_part.startswith("0/"):
        return 0, 0, status or "fnt"

    if "/" in hp_part:
        cur_s, max_s = hp_part.split("/", 1)
        try:
            cur = int(float(cur_s))
            mx = int(float(max_s))
        except ValueError:
            return 0, 0, "fnt"
        return cur, mx, status
    else:
        try:
            pct = int(float(hp_part))
            # Percentage without max — represent as pct/100
            return pct, 100, status
        except ValueError:
            return 0, 100, status


def _find_slot(slots: list[PokemonSlot], nick: str) -> Optional[PokemonSlot]:
    """Find a slot by nickname (exact match first, then case-insensitive)."""
    for s in slots:
        if s.nickname == nick:
            return s
    for s in slots:
        if s.nickname.lower() == nick.lower():
            return s
    return None


def _maybe_reveal_item(args: list[str], p1_slots: list, p2_slots: list) -> None:
    """Check for [from] item: ... annotation and reveal the item."""
    for arg in args[2:]:
        arg = arg.strip()
        if arg.startswith("[from] item:"):
            item_name = arg[len("[from] item:"):].strip()
            _reveal_source_item(args[0], item_name, p1_slots, p2_slots)
        elif arg.startswith("[of]"):
            # The item belongs to the [of] Pokemon, not the target
            pass


def _reveal_source_item(player_nick_str: str, item: str,
                        p1_slots: list, p2_slots: list) -> None:
    """Mark the item as revealed on the Pokemon that triggered it."""
    try:
        player, nick = _parse_player_nick(player_nick_str)
        slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
        if slot is not None and slot.item_revealed is None:
            slot.item_revealed = item
    except Exception:
        pass


def _maybe_reveal_ability(args: list[str], p1_slots: list, p2_slots: list) -> None:
    """Check for [from] ability: ... annotation."""
    for arg in args[1:]:
        arg = arg.strip()
        if arg.startswith("[from] ability:"):
            ability_name = arg[len("[from] ability:"):].strip()
            # The [of] annotation tells us whose ability it is
            of_str = ""
            for a2 in args[1:]:
                if a2.strip().startswith("[of]"):
                    of_str = a2.strip()[len("[of]"):].strip()
            if of_str:
                try:
                    player, nick = _parse_player_nick(of_str)
                    slot = _find_slot(p1_slots if player == 1 else p2_slots, nick)
                    if slot is not None and slot.ability_revealed is None:
                        slot.ability_revealed = ability_name
                except Exception:
                    pass


def _apply_side_start(side: SideConditions, condition: str, turn: int) -> None:
    c = condition.lower().replace(" ", "").replace("move:", "")
    if "stealthrock" in c:
        side.stealth_rock = True
    elif "spikes" in c:
        side.spikes = min(3, side.spikes + 1)
    elif "toxicspikes" in c:
        side.toxic_spikes = min(2, side.toxic_spikes + 1)
    elif "stickyweb" in c:
        side.sticky_web = True
    elif "reflect" in c:
        side.reflect = turn
        side.reflect_turn = turn
    elif "lightscreen" in c:
        side.light_screen = turn
        side.light_screen_turn = turn
    elif "auroraveil" in c:
        side.aurora_veil = turn
        side.aurora_veil_turn = turn


def _apply_side_end(side: SideConditions, condition: str) -> None:
    c = condition.lower().replace(" ", "").replace("move:", "")
    if "stealthrock" in c:
        side.stealth_rock = False
    elif "spikes" in c:
        side.spikes = 0
    elif "toxicspikes" in c:
        side.toxic_spikes = 0
    elif "stickyweb" in c:
        side.sticky_web = False
    elif "reflect" in c:
        side.reflect = 0
        side.reflect_turn = 0
    elif "lightscreen" in c:
        side.light_screen = 0
        side.light_screen_turn = 0
    elif "auroraveil" in c:
        side.aurora_veil = 0
        side.aurora_veil_turn = 0


def _parse_int_safe(s: str) -> Optional[int]:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _clone(obj):
    import copy
    return copy.deepcopy(obj)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def parse_replay_file(json_path: str, replay_id: str = "") -> Optional[ParsedBattle]:
    """Load a .json replay file and parse it."""
    import json as _json
    from pathlib import Path as _Path
    data = _json.loads(_Path(json_path).read_text(encoding="utf-8"))
    log_text = data.get("log", "")
    upload_time = data.get("uploadtime", 0) or 0
    rid = replay_id or data.get("id", "")
    parser = Gen9Parser()
    battle = parser.parse(log_text, replay_id=rid, upload_time=upload_time)
    if battle is not None:
        # Fill in ratings from search metadata if available
        if battle.p1_rating is None and data.get("rating"):
            battle.p1_rating = data["rating"]
        battle.upload_time = upload_time
    return battle
