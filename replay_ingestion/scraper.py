"""
D1: Replay scraper for Gen 9 Random Battle at 1900+ rating.

Showdown replay search API:
  GET https://replay.pokemonshowdown.com/search.json
    ?format=gen9randombattle&page=N
  Returns up to 51 entries per page, newest first.
  NOTE: the `rating` query parameter is NOT server-side filtered;
  filtering is done client-side by inspecting the `rating` field.

Individual replay JSON:
  GET https://replay.pokemonshowdown.com/{id}.json
  Returns full metadata + `log` field (pipe-delimited log text).

File layout on disk:
  data/raw_replays/gen9randombattle/{id}.json  — full replay JSON
  data/raw_replays/gen9randombattle/{id}.log   — log text (extracted from JSON)

Resumable: skips replays whose files already exist.
Rate-limited: enforces REQUEST_DELAY seconds between all HTTP requests.
"""

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SEARCH_URL  = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL  = "https://replay.pokemonshowdown.com/{replay_id}.json"
USER_AGENT  = "Mozilla/5.0 (compatible; pokemon-showdown-bot/1.0)"
REQUEST_DELAY = 1.0   # seconds between any two HTTP requests
MIN_RATING    = 1900
MAX_AGE_DAYS  = 365


class ReplayScraper:
    def __init__(
        self,
        output_dir: str,
        min_rating: int = MIN_RATING,
        max_age_days: int = MAX_AGE_DAYS,
        request_delay: float = REQUEST_DELAY,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_rating = min_rating
        self.cutoff_ts = int(
            (datetime.now(timezone.utc) - timedelta(days=max_age_days)).timestamp()
        )
        self.request_delay = request_delay
        self._last_request_time: float = 0.0

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.monotonic()

    def _fetch(self, url: str) -> bytes:
        self._rate_limit()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def _fetch_json(self, url: str) -> object:
        return json.loads(self._fetch(url))

    # ── Caching ───────────────────────────────────────────────────────────────

    def _replay_cached(self, replay_id: str) -> bool:
        return (
            (self.output_dir / f"{replay_id}.json").exists()
            and (self.output_dir / f"{replay_id}.log").exists()
        )

    def _save_replay(self, replay_id: str, search_meta: dict) -> bool:
        """Download full replay JSON, merge with search metadata, save both files."""
        try:
            full = self._fetch_json(REPLAY_URL.format(replay_id=replay_id))
        except Exception as exc:
            logger.warning("Failed to download %s: %s", replay_id, exc)
            return False

        # Merge search metadata into the replay JSON (adds rating, uploadtime if missing)
        for k, v in search_meta.items():
            full.setdefault(k, v)

        log_text: str = full.get("log", "")

        (self.output_dir / f"{replay_id}.json").write_text(
            json.dumps(full, ensure_ascii=False), encoding="utf-8"
        )
        (self.output_dir / f"{replay_id}.log").write_text(
            log_text, encoding="utf-8"
        )
        return True

    # ── Search pagination ─────────────────────────────────────────────────────

    def _search_page(self, fmt: str, page: int) -> list[dict]:
        url = f"{SEARCH_URL}?format={fmt}&page={page}"
        data = self._fetch_json(url)
        if not isinstance(data, list):
            return []
        return data

    # ── Public API ────────────────────────────────────────────────────────────

    def scrape(
        self,
        fmt: str = "gen9randombattle",
        max_replays: Optional[int] = None,
        page_limit: int = 10_000,
    ) -> dict:
        """
        Download replays for `fmt` meeting rating/age filters.

        Returns stats dict with keys: downloaded, skipped_exists,
        skipped_rating, skipped_date, skipped_private, errors, pages_fetched.
        """
        stats = {
            "downloaded": 0,
            "skipped_exists": 0,
            "skipped_rating": 0,
            "skipped_date": 0,
            "skipped_private": 0,
            "errors": 0,
            "pages_fetched": 0,
        }

        for page in range(1, page_limit + 1):
            if max_replays is not None and stats["downloaded"] >= max_replays:
                logger.info("Reached max_replays=%d, stopping.", max_replays)
                break

            logger.info("Fetching page %d (format=%s)…", page, fmt)
            try:
                entries = self._search_page(fmt, page)
            except Exception as exc:
                logger.error("Page %d fetch failed: %s", page, exc)
                stats["errors"] += 1
                break

            stats["pages_fetched"] += 1

            if not entries:
                logger.info("Empty page %d — search exhausted.", page)
                break

            reached_cutoff = False
            for entry in entries:
                if max_replays is not None and stats["downloaded"] >= max_replays:
                    break

                replay_id: str  = entry.get("id", "")
                upload_ts: int  = entry.get("uploadtime", 0) or 0
                rating: int     = entry.get("rating", 0) or 0
                private: int    = entry.get("private", 0) or 0

                if private:
                    stats["skipped_private"] += 1
                    continue
                if upload_ts < self.cutoff_ts:
                    stats["skipped_date"] += 1
                    reached_cutoff = True
                    continue
                if rating < self.min_rating:
                    stats["skipped_rating"] += 1
                    continue
                if self._replay_cached(replay_id):
                    stats["skipped_exists"] += 1
                    continue

                if self._save_replay(replay_id, entry):
                    stats["downloaded"] += 1
                    logger.info(
                        "  ✓ %s  rating=%d  %s",
                        replay_id,
                        rating,
                        datetime.fromtimestamp(upload_ts, tz=timezone.utc).date(),
                    )
                else:
                    stats["errors"] += 1

            # If the entire page is older than cutoff AND we saw old entries, stop.
            # (Page is sorted newest-first, so once we see old, rest is also old.)
            if reached_cutoff and all(
                (e.get("uploadtime") or 0) < self.cutoff_ts for e in entries
            ):
                logger.info("All entries on page %d pre-date cutoff. Stopping.", page)
                break

        return stats

    def report(self) -> dict:
        """Return counts of cached replays by date / rating band."""
        files = sorted(self.output_dir.glob("*.json"))
        total = len(files)
        ratings: list[int] = []
        for f in files:
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                r = meta.get("rating") or meta.get("players_rating")
                if isinstance(r, (int, float)):
                    ratings.append(int(r))
            except Exception:
                pass
        return {
            "total_cached": total,
            "ratings_parsed": len(ratings),
            "rating_min": min(ratings) if ratings else None,
            "rating_max": max(ratings) if ratings else None,
            "rating_mean": round(sum(ratings) / len(ratings)) if ratings else None,
        }
