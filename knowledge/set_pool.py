"""
D1: pkmn/randbats data ingestion and caching.

Gen 9 Random Battle pool is structured as:
  species → {level, roles: {role_name: {weight, moves, items, abilities, teraTypes, evs}}}

We use the stats file (has empirical frequencies) as the primary source.
The sets file (presence only, no frequencies) is fetched but not used directly.

All species-name normalization is centralized here via to_id().
"""

import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RANDBATS_URL = "https://data.pkmn.cc/randbats/gen9randombattle.json"
STATS_URL = "https://data.pkmn.cc/randbats/stats/gen9randombattle.json"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days

_DEFAULT_CACHE_DIR = Path(
    os.environ.get("METAMON_CACHE_DIR", os.path.expanduser("~/.cache/metamon"))
)

# Module-level in-memory cache (avoids repeated disk reads within a session)
_stats_data: Optional[dict] = None
_species_index: Optional[dict] = None  # normalized_id → canonical_name


# ── name normalization ─────────────────────────────────────────────────────

def to_id(name: str) -> str:
    """
    Convert any species/item/move name to Showdown's internal ID format:
    lowercase, strip all non-alphanumeric characters.

    Examples:
      "Landorus-Therian" → "landorustherian"
      "landorus therian" → "landorustherian"
      "Farfetch'd"       → "farfetchd"
      "Mr. Mime"         → "mrmime"
      "Type: Null"       → "typenull"
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ── cache / download ───────────────────────────────────────────────────────

def _needs_refresh(path: Path) -> bool:
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) > CACHE_TTL_SECONDS


def _fetch_json(url: str, cache_path: Path) -> dict:
    """Download JSON, cache to disk, return parsed dict. Falls back to stale cache on error."""
    try:
        req = urllib.request.Request(url, headers={
            "Accept-Encoding": "identity",
            "User-Agent": "Mozilla/5.0 (compatible; pokemon-showdown-bot/1.0)",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        data = json.loads(raw)
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("Fetched %s → %s", url, cache_path)
        return data
    except Exception as exc:
        if cache_path.exists():
            logger.warning("Fetch failed (%s), using stale cache %s", exc, cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"Cannot fetch {url} and no cache exists: {exc}") from exc


def _load(cache_dir: Optional[Path] = None) -> dict:
    """Load stats data, downloading if necessary. Returns the stats dict."""
    global _stats_data, _species_index
    if _stats_data is not None:
        return _stats_data

    cdir = (cache_dir or _DEFAULT_CACHE_DIR) / "randbats"
    cdir.mkdir(parents=True, exist_ok=True)

    stats_path = cdir / "gen9randombattle_stats.json"
    sets_path = cdir / "gen9randombattle_sets.json"

    # Always keep both files fresh together
    if _needs_refresh(stats_path):
        _stats_data = _fetch_json(STATS_URL, stats_path)
        # Fetch sets file too (may be useful for downstream callers)
        try:
            _fetch_json(RANDBATS_URL, sets_path)
        except Exception:
            pass
    else:
        _stats_data = json.loads(stats_path.read_text(encoding="utf-8"))

    _species_index = _build_species_index(_stats_data)
    return _stats_data


def _build_species_index(data: dict) -> dict:
    """Build {normalized_id → canonical_name} for all species."""
    index = {}
    for canonical in data.keys():
        index[to_id(canonical)] = canonical
    return index


# ── public API ─────────────────────────────────────────────────────────────

def resolve_species(name: str, cache_dir: Optional[Path] = None) -> Optional[str]:
    """
    Convert any reasonable species name to the canonical form used by pkmn/randbats.
    Returns None if the species is not in the Gen 9 Random Battle pool.

    Handles: "Landorus-Therian", "landorus-therian", "landorustherian",
             "landorus therian", "Farfetch'd", "Mr. Mime", "Type: Null"
    """
    _load(cache_dir)
    return _species_index.get(to_id(name))


def get_species_data(species: str, cache_dir: Optional[Path] = None) -> dict:
    """
    Return the full pkmn/randbats stats entry for a species.

    Return format:
      {
        "level": int,
        "roles": {
          "Role Name": {
            "weight": float,          # empirical role probability
            "moves":     {move_name: frequency, ...},
            "items":     {item_name: frequency, ...},
            "abilities": {ability_name: frequency, ...},
            "teraTypes": {type_name: frequency, ...},
            "evs":       {stat_name: int, ...},   # may be absent
          },
          ...
        }
      }

    Raises KeyError if species is not in the Gen 9 Random Battle pool.
    """
    stats = _load(cache_dir)
    canonical = resolve_species(species, cache_dir)
    if canonical is None:
        raise KeyError(f"{species!r} not found in Gen 9 Random Battle pool")
    return stats[canonical]


def get_all_species(cache_dir: Optional[Path] = None) -> list[str]:
    """Return sorted list of all canonical species names in the pool."""
    stats = _load(cache_dir)
    return sorted(stats.keys())


def get_pool_size(cache_dir: Optional[Path] = None) -> int:
    return len(_load(cache_dir))


def verify_frequencies(species: str, cache_dir: Optional[Path] = None) -> bool:
    """Return True if role weights for this species sum to ~1.0."""
    data = get_species_data(species, cache_dir)
    total = sum(r.get("weight", 0.0) for r in data["roles"].values())
    return abs(total - 1.0) < 1e-3
