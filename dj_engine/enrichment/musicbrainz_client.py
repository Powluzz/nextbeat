"""MusicBrainz-verrijking: MBID + genre-tags opzoeken op titel/artiest.

Verplichte MusicBrainz-gebruiksvoorwaarden waar deze module zich aan houdt:
- Max. 1 request/seconde (RateLimiter hieronder, plus musicbrainzngs' eigen
  interne rate limiter als tweede vangnet).
- Een unieke, identificeerbare User-Agent (naam/versie/contact), ingesteld
  via config.yaml (musicbrainz.user_agent_*) — nooit hardcoded.

Resultaten worden gecached in de `musicbrainz_cache`-tabel (zie db.py) zodat
een herhaalde ingest van dezelfde track geen nieuwe request doet.
Netwerkfouten/timeouts crashen de lookup nooit — na `max_retries` pogingen
wordt (None, None) teruggegeven en gelogd, zodat de ingest-pipeline doorgaat.
"""
from __future__ import annotations

import logging
import socket
import time
from typing import Any

from dj_engine import db as db_module

logger = logging.getLogger(__name__)


class RateLimiter:
    """Eenvoudige, testbare throttle: garandeert minimaal `1/requests_per_second`
    seconden tussen opeenvolgende `wait()`-aanroepen.

    Los van musicbrainzngs' eigen rate limiter geïmplementeerd, zodat het
    gedrag direct unit-testbaar is (met injecteerbare time/sleep-functies)
    zonder afhankelijk te zijn van een externe library-implementatie.
    """

    def __init__(
        self,
        requests_per_second: float,
        sleep_fn=time.sleep,
        time_fn=time.monotonic,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second moet > 0 zijn")
        self.min_interval = 1.0 / requests_per_second
        self._sleep = sleep_fn
        self._now = time_fn
        self._last_call: float | None = None

    def wait(self) -> None:
        """Blokkeer (indien nodig) tot het veilig is om de volgende request te doen."""
        now = self._now()
        if self._last_call is not None:
            elapsed = now - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
        self._last_call = self._now()


class MusicBrainzClient:
    """Client voor MBID + genre-tag lookups via de MusicBrainz webservice."""

    def __init__(
        self,
        config: dict[str, Any],
        db_conn=None,
        mb_module=None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """
        Args:
            config: volledige app-config (zoals geladen door dj_engine.config.load_config).
            db_conn: optionele sqlite3-connectie voor caching. Zonder connectie
                wordt niet gecached (elke lookup gaat naar het netwerk).
            mb_module: override voor musicbrainzngs (voor tests); default is
                de echte library, lazy geïmporteerd.
            rate_limiter: override voor de throttle (voor tests).
        """
        self.config = config["musicbrainz"]
        self.db_conn = db_conn

        if mb_module is None:
            import musicbrainzngs as mb_module  # lazy: netwerk-dependency
        self._mb = mb_module

        self.rate_limiter = rate_limiter or RateLimiter(
            self.config["rate_limit_per_second"]
        )
        self._configured = False

    def _ensure_configured(self) -> None:
        if self._configured:
            return

        contact = self.config.get("user_agent_contact") or ""
        if not contact:
            logger.warning(
                "musicbrainz.user_agent_contact is niet ingesteld in config.yaml — "
                "MusicBrainz vereist een identificeerbaar contactadres in de User-Agent."
            )

        self._mb.set_useragent(
            self.config["user_agent_name"],
            self.config["user_agent_version"],
            contact,
        )
        # Interne rate limiter van musicbrainzngs als extra vangnet naast onze
        # eigen RateLimiter (1 request per `interval` seconden).
        rate = self.config["rate_limit_per_second"]
        self._mb.set_rate_limit(limit_or_interval=1.0 / rate, new_requests=1)

        timeout = self.config.get("timeout_seconds")
        if timeout:
            # musicbrainzngs biedt geen per-call timeout-parameter; het gebruikt
            # urllib met de globale socket-default. Dit proces-brede default is
            # voor dit doel (een lokale CLI-tool) acceptabel.
            socket.setdefaulttimeout(timeout)

        self._configured = True

    @staticmethod
    def _cache_key(title: str, artist: str) -> str:
        return f"{artist.strip().lower()}|{title.strip().lower()}"

    def lookup_track(
        self, title: str | None, artist: str | None
    ) -> tuple[str | None, str | None]:
        """Zoek MBID en genre-tags op voor (title, artist).

        Returns:
            (mbid, genre_tags) — genre_tags is een komma-gescheiden string,
            of None als er geen match/tags gevonden zijn. Geeft (None, None)
            terug bij ontbrekende input of na uitgeputte retries op
            netwerkfouten — nooit een gefabriceerd resultaat.
        """
        if not title or not artist:
            return None, None

        cache_key = self._cache_key(title, artist)
        if self.db_conn is not None:
            cached = db_module.get_cached_lookup(self.db_conn, cache_key)
            if cached is not None:
                return cached["mbid"], cached["genre_tags"]

        self._ensure_configured()

        max_retries = self.config.get("max_retries", 3)
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            self.rate_limiter.wait()
            try:
                result = self._mb.search_recordings(recording=title, artist=artist, limit=1)
                mbid = self._extract_mbid(result)

                genre_tags = None
                if mbid is not None:
                    # search_recordings() levert geen tag-list mee (de MB
                    # search-API ondersteunt geen includes) — een tweede,
                    # eveneens rate-limited call is nodig voor genre-tags.
                    self.rate_limiter.wait()
                    detail = self._mb.get_recording_by_id(mbid, includes=["tags"])
                    genre_tags = self._extract_tags(detail.get("recording", {}))

                if self.db_conn is not None:
                    db_module.set_cached_lookup(self.db_conn, cache_key, mbid, genre_tags)
                return mbid, genre_tags
            except self._mb.MusicBrainzError as exc:
                last_error = exc
                logger.warning(
                    "MusicBrainz lookup mislukt (poging %d/%d) voor '%s - %s': %s",
                    attempt, max_retries, artist, title, exc,
                )
            except (OSError, socket.timeout) as exc:
                # Netwerk-/timeoutfouten die niet via MusicBrainzError lopen.
                last_error = exc
                logger.warning(
                    "Netwerkfout (poging %d/%d) bij MusicBrainz-lookup voor '%s - %s': %s",
                    attempt, max_retries, artist, title, exc,
                )

        logger.error(
            "MusicBrainz lookup definitief mislukt voor '%s - %s' na %d pogingen: %s",
            artist, title, max_retries, last_error,
        )
        return None, None

    @staticmethod
    def _extract_mbid(search_result: dict[str, Any]) -> str | None:
        recordings = search_result.get("recording-list") or []
        if not recordings:
            return None
        return recordings[0].get("id")

    @staticmethod
    def _extract_tags(recording: dict[str, Any]) -> str | None:
        tags = recording.get("tag-list") or []
        names = [t["name"] for t in tags if isinstance(t, dict) and t.get("name")]
        return ",".join(names) if names else None
