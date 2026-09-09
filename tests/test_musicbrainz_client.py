"""Tests voor dj_engine.enrichment.musicbrainz_client.

Alle netwerkcalls zijn gemockt (zoals BUILD_SPEC.md vereist) — er gaat
nooit een echt HTTP-verzoek naar musicbrainz.org uit tijdens tests.
"""
from __future__ import annotations

import time

import pytest

from dj_engine import db
from dj_engine.enrichment.musicbrainz_client import MusicBrainzClient, RateLimiter


# --- RateLimiter ----------------------------------------------------------

def test_rate_limiter_first_call_does_not_sleep():
    sleeps = []
    limiter = RateLimiter(1.0, sleep_fn=sleeps.append, time_fn=lambda: 100.0)
    limiter.wait()
    assert sleeps == []


def test_rate_limiter_enforces_minimum_interval():
    sleeps = []
    clock = {"t": 0.0}

    def fake_time():
        return clock["t"]

    limiter = RateLimiter(1.0, sleep_fn=sleeps.append, time_fn=fake_time)
    limiter.wait()  # t=0.0, geen sleep
    clock["t"] = 0.3  # slechts 0.3s later
    limiter.wait()
    assert sleeps == [pytest.approx(0.7)]  # moet nog 0.7s wachten tot 1.0s


def test_rate_limiter_no_sleep_if_enough_time_passed():
    sleeps = []
    clock = {"t": 0.0}
    limiter = RateLimiter(1.0, sleep_fn=sleeps.append, time_fn=lambda: clock["t"])
    limiter.wait()
    clock["t"] = 5.0
    limiter.wait()
    assert sleeps == []


def test_rate_limiter_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        RateLimiter(0)
    with pytest.raises(ValueError):
        RateLimiter(-1)


def test_rate_limiter_real_time_respects_one_per_second():
    """Integratietest met echte klok (geen netwerk): twee wait()-calls
    achter elkaar moeten samen >= ~1s duren bij 1 req/s."""
    limiter = RateLimiter(1.0)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.9


# --- MusicBrainzClient ------------------------------------------------

class FakeMusicBrainzError(Exception):
    pass


class FakeMBModule:
    """Minimale stand-in voor musicbrainzngs, met scriptbare responses.

    Bootst de echte API na: search_recordings() levert GEEN tag-list mee
    (de MB search-API ondersteunt geen includes) — tags komen alleen via
    een tweede get_recording_by_id(mbid, includes=["tags"])-call, precies
    zoals bij de echte MusicBrainz-webservice.
    """

    MusicBrainzError = FakeMusicBrainzError

    def __init__(self, search_responses=None, tag_lists=None, raise_times=0):
        self.search_responses = search_responses if search_responses is not None else []
        self.tag_lists = tag_lists or {}  # mbid -> lijst van {"name": ...} dicts
        self.raise_times = raise_times
        self.call_count = 0
        self.detail_call_count = 0
        self.useragent_calls = []
        self.rate_limit_calls = []

    def set_useragent(self, app, version, contact):
        self.useragent_calls.append((app, version, contact))

    def set_rate_limit(self, limit_or_interval, new_requests):
        self.rate_limit_calls.append((limit_or_interval, new_requests))

    def search_recordings(self, recording, artist, limit=1):
        self.call_count += 1
        if self.call_count <= self.raise_times:
            raise self.MusicBrainzError("simulated network error")
        if self.search_responses:
            return self.search_responses.pop(0)
        return {"recording-list": []}

    def get_recording_by_id(self, mbid, includes=None):
        self.call_count += 1
        self.detail_call_count += 1
        if self.call_count <= self.raise_times:
            raise self.MusicBrainzError("simulated network error")
        return {"recording": {"id": mbid, "tag-list": self.tag_lists.get(mbid)}}


def _config():
    return {
        "musicbrainz": {
            "rate_limit_per_second": 1000.0,  # tests niet laten wachten
            "user_agent_name": "dj-engine-test",
            "user_agent_version": "0.0.1",
            "user_agent_contact": "test@example.invalid",
            "max_retries": 3,
            "timeout_seconds": 5,
        }
    }


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def test_lookup_track_returns_mbid_and_genres(conn):
    fake_mb = FakeMBModule(
        search_responses=[{"recording-list": [{"id": "abc-123"}]}],
        tag_lists={"abc-123": [{"name": "techno"}, {"name": "house"}]},
    )
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)

    mbid, genres = client.lookup_track("Strings of Life", "Derrick May")

    assert mbid == "abc-123"
    assert genres == "techno,house"
    assert fake_mb.detail_call_count == 1  # tweede call nodig voor tags
    assert fake_mb.useragent_calls == [("dj-engine-test", "0.0.1", "test@example.invalid")]


def test_lookup_track_no_match_returns_none_none(conn):
    fake_mb = FakeMBModule(search_responses=[{"recording-list": []}])
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)

    mbid, genres = client.lookup_track("Onbekend Nummer", "Onbekende Artiest")

    assert (mbid, genres) is not None
    assert mbid is None
    assert genres is None
    assert fake_mb.detail_call_count == 0  # geen mbid -> geen detail-call nodig


def test_lookup_track_missing_title_or_artist_skips_network(conn):
    fake_mb = FakeMBModule()
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)

    assert client.lookup_track(None, "Artist") == (None, None)
    assert client.lookup_track("Title", None) == (None, None)
    assert fake_mb.call_count == 0


def test_lookup_track_uses_cache_on_second_call(conn):
    fake_mb = FakeMBModule(search_responses=[{"recording-list": [{"id": "xyz"}]}])
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)

    first = client.lookup_track("Title", "Artist")
    second = client.lookup_track("Title", "Artist")

    assert first == second == ("xyz", None)
    # Eerste lookup: 1x search + 1x detail (voor tags) = 2 calls.
    # Tweede lookup kwam uit cache, geen extra netwerkcalls.
    assert fake_mb.call_count == 2


def test_lookup_track_retries_on_network_error_then_succeeds(conn):
    fake_mb = FakeMBModule(
        search_responses=[{"recording-list": [{"id": "retry-ok"}]}],
        raise_times=2,
    )
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)

    mbid, genres = client.lookup_track("Title", "Artist")

    assert mbid == "retry-ok"
    # 2x mislukte search-poging, 3e search gelukt, daarna 1x detail-call = 4 calls.
    assert fake_mb.call_count == 4


def test_lookup_track_gives_up_after_max_retries_without_crashing(conn):
    fake_mb = FakeMBModule(search_responses=[], raise_times=99)
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)

    # Mag geen exception opgooien, ook al blijven alle pogingen falen.
    mbid, genres = client.lookup_track("Title", "Artist")

    assert (mbid, genres) == (None, None)
    assert fake_mb.call_count == 3  # max_retries uit config


def test_lookup_track_failed_lookup_is_not_cached(conn):
    """Een netwerkfout mag niet gecached worden als 'geen match' — anders
    wordt een tijdelijke storing permanent als lege data vastgelegd."""
    fake_mb = FakeMBModule(search_responses=[], raise_times=99)
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)

    client.lookup_track("Title", "Artist")
    assert db.get_cached_lookup(conn, "artist|title") is None


def test_lookup_track_without_db_conn_does_not_cache():
    fake_mb = FakeMBModule(
        search_responses=[{"recording-list": [{"id": "no-cache"}]}] * 2,
        tag_lists={"no-cache": []},
    )
    client = MusicBrainzClient(_config(), db_conn=None, mb_module=fake_mb)

    client.lookup_track("Title", "Artist")
    client.lookup_track("Title", "Artist")

    # Geen cache -> elke lookup doet opnieuw search + detail = 2x2 = 4 calls.
    assert fake_mb.call_count == 4


def test_lookup_track_search_response_without_tags_still_fetches_detail(conn):
    """Regressietest: search_recordings() levert in de echte MB-API nooit
    tag-list mee (bevestigd tegen de echte webservice) — de client moet
    hierdoor niet stilzwijgend genre_tags=None teruggeven zonder ooit de
    detail-call te proberen."""
    fake_mb = FakeMBModule(
        search_responses=[{"recording-list": [{"id": "real-world-shape"}]}],
        tag_lists={"real-world-shape": [{"name": "electro"}, {"name": "house"}]},
    )
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)

    mbid, genres = client.lookup_track("Strobe", "Deadmau5")

    assert mbid == "real-world-shape"
    assert genres == "electro,house"
    assert fake_mb.detail_call_count == 1


def test_lookup_track_no_tags_available_returns_none_genre(conn):
    """Sommige recordings hebben simpelweg geen MB-tags (echt geobserveerd
    gedrag) — dat moet resulteren in genre=None, niet een fout."""
    fake_mb = FakeMBModule(
        search_responses=[{"recording-list": [{"id": "no-tags-mbid"}]}],
        tag_lists={},  # geen entry -> tag-list is None
    )
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)

    mbid, genres = client.lookup_track("Hey Brother", "Avicii")

    assert mbid == "no-tags-mbid"
    assert genres is None


def test_client_sets_useragent_before_first_call(conn):
    fake_mb = FakeMBModule(search_responses=[{"recording-list": []}])
    client = MusicBrainzClient(_config(), db_conn=conn, mb_module=fake_mb)
    assert fake_mb.useragent_calls == []  # nog niet geconfigureerd
    client.lookup_track("Title", "Artist")
    assert len(fake_mb.useragent_calls) == 1
