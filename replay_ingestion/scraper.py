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
Backoff: exponential retry on HTTP errors (429, 5xx) up to BACKOFF_MAX_WAIT seconds.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SEARCH_URL     = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL     = "https://replay.pokemonshowdown.com/{replay_id}.json"
USER_AGENT     = "Mozilla/5.0 (compatible; pokemon-showdown-bot/1.0)"
REQUEST_DELAY  = 1.0   # seconds between any two HTTP requests
MIN_RATING     = 1900
MAX_AGE_DAYS   = 365
BACKOFF_BASE   = 5.0   # initial backoff on retryable error (seconds)
BACKOFF_MAX    = 60.0  # cap for exponential backoff (seconds)
BACKOFF_RETRIES = 5    # max retries before giving up on a single request
PROGRESS_EVERY = 100   # log a progress line every N downloads

# Species that use Illusion ability (lowercased Showdown IDs)
_ILLUSION_SPECIES = {"zoroark", "zoroarkhisui", "zorua", "zoruahisui"}


def _has_illusion_team(log_text: str) -> bool:
    """Return True if any |switch| or |drag| line names a Zoroark/Zorua species."""
    for line in log_text.splitlines():
        if not (line.startswith("|switch|") or line.startswith("|drag|")):
            continue
        # |switch|p1a: Nick|Species, L86, M|HP
        parts = line.split("|")
        if len(parts) >= 4:
            detail = parts[3].split(",")[0].strip().lower().replace("-", "").replace(" ", "")
            if detail in _ILLUSION_SPECIES:
                return True
    return False


def _has_illusion_break(log_text: str) -> bool:
    """Return True if an actual |-end|...|Illusion event appears in the log."""
    return bool(re.search(r"\|-end\|[^|]+\|Illusion", log_text))


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

    def _fetch(self, url: str, retryable: bool = True) -> bytes:
        """Fetch URL with rate-limiting and exponential backoff on errors."""
        wait = BACKOFF_BASE
        for attempt in range(BACKOFF_RETRIES if retryable else 1):
            self._rate_limit()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and retryable:
                    logger.warning(
                        "HTTP %d on %s — backing off %.0fs (attempt %d/%d)",
                        exc.code, url, wait, attempt + 1, BACKOFF_RETRIES,
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, BACKOFF_MAX)
                else:
                    raise
            except OSError as exc:
                if retryable and attempt < BACKOFF_RETRIES - 1:
                    logger.warning(
                        "Network error on %s: %s — backing off %.0fs",
                        url, exc, wait,
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, BACKOFF_MAX)
                else:
                    raise
        raise RuntimeError(f"Exhausted {BACKOFF_RETRIES} retries for {url}")

    def _fetch_json(self, url: str, retryable: bool = True) -> object:
        return json.loads(self._fetch(url, retryable=retryable))

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
        try:
            data = self._fetch_json(url)
        except Exception:
            raise
        if not isinstance(data, list):
            return []
        return data

    # ── Public API ────────────────────────────────────────────────────────────

    def scrape(
        self,
        fmt: str = "gen9randombattle",
        max_replays: Optional[int] = None,
        page_limit: int = 10_000,
        progress_every: int = PROGRESS_EVERY,
    ) -> dict:
        """
        Download replays for `fmt` meeting rating/age filters.

        Returns stats dict including Illusion monitoring counters:
          illusion_team  — replays where Zoroark/Zorua appeared in any team
          illusion_break — replays where Illusion actually broke mid-battle
        """
        stats = {
            "downloaded": 0,
            "skipped_exists": 0,
            "skipped_rating": 0,
            "skipped_date": 0,
            "skipped_private": 0,
            "errors": 0,
            "pages_fetched": 0,
            "illusion_team": 0,
            "illusion_break": 0,
        }
        start_time = time.monotonic()

        for page in range(1, page_limit + 1):
            if max_replays is not None and stats["downloaded"] >= max_replays:
                logger.info("Reached max_replays=%d, stopping.", max_replays)
                break

            logger.debug("Fetching page %d (format=%s)…", page, fmt)
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

                    # Illusion monitoring
                    log_path = self.output_dir / f"{replay_id}.log"
                    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
                    if _has_illusion_team(log_text):
                        stats["illusion_team"] += 1
                        if _has_illusion_break(log_text):
                            stats["illusion_break"] += 1

                    if stats["downloaded"] % progress_every == 0:
                        elapsed = time.monotonic() - start_time
                        rate = stats["downloaded"] / elapsed if elapsed > 0 else 0
                        logger.info(
                            "PROGRESS: %d replays downloaded | +%d exist | "
                            "%d errors | %.1f/min | illusion_team=%d illusion_break=%d",
                            stats["downloaded"],
                            stats["skipped_exists"],
                            stats["errors"],
                            rate * 60,
                            stats["illusion_team"],
                            stats["illusion_break"],
                        )
                    else:
                        logger.debug(
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
        """Return counts of cached replays and Illusion statistics."""
        files = sorted(self.output_dir.glob("*.json"))
        total = len(files)
        ratings: list[int] = []
        illusion_team = 0
        illusion_break = 0
        for f in files:
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                r = meta.get("rating") or meta.get("players_rating")
                if isinstance(r, (int, float)):
                    ratings.append(int(r))
            except Exception:
                pass
            log_path = f.with_suffix(".log")
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8")
                if _has_illusion_team(log_text):
                    illusion_team += 1
                    if _has_illusion_break(log_text):
                        illusion_break += 1
        return {
            "total_cached": total,
            "ratings_parsed": len(ratings),
            "rating_min": min(ratings) if ratings else None,
            "rating_max": max(ratings) if ratings else None,
            "rating_mean": round(sum(ratings) / len(ratings)) if ratings else None,
            "illusion_team": illusion_team,
            "illusion_break": illusion_break,
        }
