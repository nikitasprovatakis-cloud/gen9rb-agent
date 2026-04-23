"""
Player-based supplement scraper for Phase 3.

Scrapes gen9randombattle replays for a list of specific players, applying:
  - Date filter: uploadtime > SUPPLEMENT_CUTOFF (July 23, 2025 00:00:00 UTC)
  - Rating filter: >= 1900 (or any smogtours-* replay regardless of rating)
  - Per-player cap: 500 replays
  - Total cap: 5000 replays
  - Within-supplement dedup: shared replays credited to all participating players
  - Against-HolidayOugi dedup: replays already in HolidayOugi are skipped

Metadata added to each supplementary replay JSON:
  dataset: "main"
  source: "player_supplement"
  supplementary_players: [list of matching normalized player IDs]
  scraped_date: <unix timestamp when downloaded>
  upload_date: <replay's uploadtime>

File layout:
  data/raw_replays/supplement/{replay_id}.json
  data/raw_replays/supplement/{replay_id}.log  (log text extracted)

Resume-safe: skips replay IDs already on disk.
Rate limit: 1 second between all HTTP requests.
Backoff: exponential on 429/5xx, same parameters as main scraper.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SEARCH_URL      = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL      = "https://replay.pokemonshowdown.com/{replay_id}.json"
USER_AGENT      = "Mozilla/5.0 (compatible; pokemon-showdown-bot/1.0)"
REQUEST_DELAY   = 1.0
BACKOFF_BASE    = 5.0
BACKOFF_MAX     = 60.0
BACKOFF_RETRIES = 5

# April 22, 2026 00:00:00 UTC — HolidayOugi covers through 2026-04-21
SUPPLEMENT_CUTOFF = 1776873600  # 2026-04-22T00:00:00Z

PER_PLAYER_CAP = 500
TOTAL_CAP      = 5_000
MIN_RATING     = 1900


def normalize_id(name: str) -> str:
    """Showdown ID: lowercase, strip all non-alphanumeric chars."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


class PlayerScraper:
    def __init__(
        self,
        output_dir: Path,
        holidayougi_ids: Optional[set] = None,
        request_delay: float = REQUEST_DELAY,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.holidayougi_ids: set = holidayougi_ids or set()
        self.request_delay = request_delay
        self._last_request: float = 0.0

        # Shared replay dedup: id → set of player IDs that matched it
        self._seen_ids: dict[str, set] = {}

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()

    def _fetch(self, url: str) -> bytes:
        wait = BACKOFF_BASE
        for attempt in range(BACKOFF_RETRIES):
            self._rate_limit()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504):
                    logger.warning("HTTP %d on %s — backing off %.0fs (attempt %d/%d)",
                                   exc.code, url, wait, attempt + 1, BACKOFF_RETRIES)
                    time.sleep(wait)
                    wait = min(wait * 2, BACKOFF_MAX)
                else:
                    raise
            except OSError as exc:
                if attempt < BACKOFF_RETRIES - 1:
                    logger.warning("Network error on %s: %s — backing off %.0fs", url, exc, wait)
                    time.sleep(wait)
                    wait = min(wait * 2, BACKOFF_MAX)
                else:
                    raise
        raise RuntimeError(f"Exhausted {BACKOFF_RETRIES} retries for {url}")

    def _fetch_json(self, url: str) -> object:
        return json.loads(self._fetch(url))

    # ── Existence checks ──────────────────────────────────────────────────────

    def _on_disk(self, replay_id: str) -> bool:
        return (self.output_dir / f"{replay_id}.json").exists()

    def _in_holidayougi(self, replay_id: str) -> bool:
        return replay_id in self.holidayougi_ids

    # ── Per-player pagination ─────────────────────────────────────────────────

    def _player_search(self, player_id: str, before: Optional[int] = None) -> list[dict]:
        url = f"{SEARCH_URL}?user={player_id}&format=gen9randombattle"
        if before is not None:
            url += f"&before={before}"
        try:
            data = self._fetch_json(url)
        except Exception as exc:
            logger.error("Search failed for %s: %s", player_id, exc)
            return []
        return data if isinstance(data, list) else []

    def _qualifies(self, entry: dict) -> bool:
        """True if this search-result entry should be downloaded."""
        upload_ts = entry.get("uploadtime", 0) or 0
        rating    = entry.get("rating", 0) or 0
        private   = entry.get("private", 0) or 0
        replay_id = entry.get("id", "")

        if private:
            return False
        if upload_ts <= SUPPLEMENT_CUTOFF:
            return False  # before cutoff or at exact boundary

        # Tournament replays always qualify if uploadtime passes
        if replay_id.startswith("smogtours-"):
            return True

        return rating >= MIN_RATING

    def scrape_player(
        self,
        player_id: str,
        norm_id: str,
        total_downloaded: int,
    ) -> dict:
        """
        Paginate through one player's replays after SUPPLEMENT_CUTOFF.
        Returns stats dict: {downloaded, skipped_rating, skipped_date, skipped_exists,
                             skipped_holidayougi, shared, errors}
        """
        stats = dict(downloaded=0, skipped_rating=0, skipped_date=0,
                     skipped_exists=0, skipped_holidayougi=0, shared=0, errors=0)
        player_count = 0
        before: Optional[int] = None

        while player_count < PER_PLAYER_CAP and (total_downloaded + stats["downloaded"]) < TOTAL_CAP:
            entries = self._player_search(norm_id, before=before)
            if not entries:
                break

            oldest_ts = None
            stop_pagination = False

            for entry in entries:
                upload_ts  = entry.get("uploadtime", 0) or 0
                replay_id  = entry.get("id", "")
                rating     = entry.get("rating", 0) or 0

                if oldest_ts is None or upload_ts < oldest_ts:
                    oldest_ts = upload_ts

                # Stop paginating once we're past the cutoff
                if upload_ts <= SUPPLEMENT_CUTOFF:
                    stop_pagination = True
                    stats["skipped_date"] += 1
                    continue

                if not self._qualifies(entry):
                    stats["skipped_rating"] += 1
                    continue

                if self._in_holidayougi(replay_id):
                    stats["skipped_holidayougi"] += 1
                    logger.debug("Skipped (in HolidayOugi): %s", replay_id)
                    continue

                # Already in supplement from another player?
                if replay_id in self._seen_ids:
                    self._seen_ids[replay_id].add(norm_id)
                    stats["shared"] += 1
                    # Update supplementary_players in the saved JSON
                    self._update_players_field(replay_id, norm_id)
                    continue

                if self._on_disk(replay_id):
                    # Downloaded in a previous run — register in seen_ids
                    self._seen_ids[replay_id] = {norm_id}
                    stats["skipped_exists"] += 1
                    continue

                # Download and save
                if self._download_replay(replay_id, entry, norm_id):
                    self._seen_ids[replay_id] = {norm_id}
                    stats["downloaded"] += 1
                    player_count += 1
                    n_total = total_downloaded + stats["downloaded"]
                    if n_total % 100 == 0:
                        logger.info(
                            "PROGRESS: supplement total=%d | player=%s count=%d",
                            n_total, player_id, player_count,
                        )
                    if player_count >= PER_PLAYER_CAP or n_total >= TOTAL_CAP:
                        break
                else:
                    stats["errors"] += 1

            if stop_pagination or oldest_ts is None or oldest_ts <= SUPPLEMENT_CUTOFF:
                break
            before = oldest_ts

        return stats

    def _download_replay(self, replay_id: str, search_meta: dict, norm_id: str) -> bool:
        try:
            full = self._fetch_json(REPLAY_URL.format(replay_id=replay_id))
        except Exception as exc:
            logger.warning("Failed to download %s: %s", replay_id, exc)
            return False

        for k, v in search_meta.items():
            full.setdefault(k, v)

        full["dataset"]               = "main"
        full["source"]                = "player_supplement"
        full["supplementary_players"] = [norm_id]
        full["scraped_date"]          = int(time.time())
        full["upload_date"]           = full.get("uploadtime", 0)

        log_text = full.get("log", "")
        (self.output_dir / f"{replay_id}.json").write_text(
            json.dumps(full, ensure_ascii=False), encoding="utf-8"
        )
        (self.output_dir / f"{replay_id}.log").write_text(log_text, encoding="utf-8")
        logger.debug("Saved %s (player=%s rating=%s)", replay_id, norm_id,
                     full.get("rating", "?"))
        return True

    def _update_players_field(self, replay_id: str, norm_id: str) -> None:
        """Add norm_id to supplementary_players in an already-saved JSON."""
        p = self.output_dir / f"{replay_id}.json"
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            players = data.get("supplementary_players", [])
            if norm_id not in players:
                players.append(norm_id)
                data["supplementary_players"] = players
                p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not update players field for %s: %s", replay_id, exc)

    # ── Multi-player orchestration ────────────────────────────────────────────

    def scrape_all(self, players: list[tuple[str, str]]) -> dict:
        """
        players: list of (display_name, normalized_id)
        Returns aggregate stats and per-player breakdown.
        """
        aggregate = dict(downloaded=0, shared=0, errors=0,
                         skipped_rating=0, skipped_date=0,
                         skipped_exists=0, skipped_holidayougi=0)
        per_player = {}
        total_downloaded = 0

        for display, norm_id in players:
            if total_downloaded >= TOTAL_CAP:
                logger.info("Total cap (%d) reached. Stopping.", TOTAL_CAP)
                break
            logger.info("Scraping player: %s (%s)", display, norm_id)
            stats = self.scrape_player(display, norm_id, total_downloaded)
            per_player[display] = stats
            total_downloaded += stats["downloaded"]
            for k in aggregate:
                aggregate[k] += stats.get(k, 0)
            logger.info(
                "Player %s done: downloaded=%d shared=%d skipped_rating=%d "
                "skipped_date=%d errors=%d",
                display, stats["downloaded"], stats["shared"],
                stats["skipped_rating"], stats["skipped_date"], stats["errors"],
            )

        aggregate["total_unique_replays"] = total_downloaded
        aggregate["per_player"] = per_player
        return aggregate

    # ── HolidayOugi ID loader ─────────────────────────────────────────────────

    @staticmethod
    def load_holidayougi_ids(parquet_dir: Path) -> set:
        """Load all replay IDs from the HolidayOugi parquet files into a set."""
        import pandas as pd
        ids: set = set()
        for part in sorted(parquet_dir.glob("part*.parquet")):
            df = pd.read_parquet(part, columns=["id"])
            ids.update(df["id"].tolist())
            logger.info("Loaded %d IDs from %s (running total: %d)", len(df), part.name, len(ids))
        return ids

    # ── Account verification ──────────────────────────────────────────────────

    def verify_player(self, norm_id: str) -> bool:
        """Return True if the player has at least one gen9randombattle replay."""
        url = f"{SEARCH_URL}?user={norm_id}&format=gen9randombattle"
        try:
            data = self._fetch_json(url)
            return isinstance(data, list) and len(data) > 0
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise
        except Exception:
            return False
