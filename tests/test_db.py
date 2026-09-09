"""Tests voor dj_engine.db: schema + CRUD."""
from __future__ import annotations

import sqlite3

import pytest

from dj_engine import db


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def _sample_track(**overrides) -> dict:
    track = {
        "filepath": "/music/track1.mp3",
        "title": "Test Track",
        "artist": "Test Artist",
        "bpm": 128.0,
        "key": "C",
        "scale": "major",
        "camelot": "8B",
        "energy": 0.8,
    }
    track.update(overrides)
    return track


def test_schema_creates_tables(conn):
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"tracks", "transitions", "musicbrainz_cache"} <= tables


def test_insert_and_get_track(conn):
    track_id = db.insert_track(conn, _sample_track())
    assert track_id == 1

    row = db.get_track(conn, track_id)
    assert row["filepath"] == "/music/track1.mp3"
    assert row["bpm"] == 128.0
    assert row["camelot"] == "8B"
    # Niet-opgegeven velden mogen nooit gefabriceerd worden
    assert row["mood_happy"] is None
    assert row["genre"] is None
    assert row["analyzed_at"] is not None


def test_get_track_missing_returns_none(conn):
    assert db.get_track(conn, 999) is None


def test_track_exists(conn):
    assert db.track_exists(conn, "/music/track1.mp3") is False
    db.insert_track(conn, _sample_track())
    assert db.track_exists(conn, "/music/track1.mp3") is True


def test_insert_duplicate_filepath_raises(conn):
    db.insert_track(conn, _sample_track())
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_track(conn, _sample_track())


def test_upsert_track_is_idempotent(conn):
    track_id_1 = db.upsert_track(conn, _sample_track(bpm=128.0))
    track_id_2 = db.upsert_track(conn, _sample_track(bpm=130.0))

    assert track_id_1 == track_id_2
    assert len(db.get_all_tracks(conn)) == 1
    assert db.get_track(conn, track_id_1)["bpm"] == 130.0


def test_get_all_tracks_ordered(conn):
    db.insert_track(conn, _sample_track(filepath="/music/a.mp3"))
    db.insert_track(conn, _sample_track(filepath="/music/b.mp3"))
    tracks = db.get_all_tracks(conn)
    assert [t["filepath"] for t in tracks] == ["/music/a.mp3", "/music/b.mp3"]


def test_search_tracks_by_artist(conn):
    db.insert_track(conn, _sample_track(filepath="/a.mp3", artist="Daft Punk"))
    db.insert_track(conn, _sample_track(filepath="/b.mp3", artist="Justice"))
    results = db.search_tracks(conn, artist="daft")
    assert len(results) == 1
    assert results[0]["artist"] == "Daft Punk"


def test_search_tracks_by_bpm_range(conn):
    db.insert_track(conn, _sample_track(filepath="/a.mp3", bpm=120))
    db.insert_track(conn, _sample_track(filepath="/b.mp3", bpm=128))
    db.insert_track(conn, _sample_track(filepath="/c.mp3", bpm=140))
    results = db.search_tracks(conn, bpm_min=122, bpm_max=132)
    assert [t["filepath"] for t in results] == ["/b.mp3"]


def test_search_tracks_by_genre(conn):
    db.insert_track(conn, _sample_track(filepath="/a.mp3", genre="Techno"))
    db.insert_track(conn, _sample_track(filepath="/b.mp3", genre="House"))
    results = db.search_tracks(conn, genre="techno")
    assert len(results) == 1


def test_delete_track(conn):
    track_id = db.insert_track(conn, _sample_track())
    assert db.delete_track(conn, track_id) is True
    assert db.get_track(conn, track_id) is None
    assert db.delete_track(conn, track_id) is False


def test_log_and_get_transitions(conn):
    id_a = db.insert_track(conn, _sample_track(filepath="/a.mp3"))
    id_b = db.insert_track(conn, _sample_track(filepath="/b.mp3"))

    transition_id = db.log_transition(conn, id_a, id_b, "build", rating=4)
    assert transition_id == 1

    transitions = db.get_transitions(conn)
    assert len(transitions) == 1
    assert transitions[0]["direction"] == "build"
    assert transitions[0]["rating"] == 4


def test_get_recent_track_ids(conn):
    id_a = db.insert_track(conn, _sample_track(filepath="/a.mp3"))
    id_b = db.insert_track(conn, _sample_track(filepath="/b.mp3"))
    id_c = db.insert_track(conn, _sample_track(filepath="/c.mp3"))

    db.log_transition(conn, id_a, id_b, "build")
    db.log_transition(conn, id_b, id_c, "hold")

    recent = db.get_recent_track_ids(conn, limit=5)
    assert recent == [id_c, id_b]


def test_get_recent_track_ids_zero_limit(conn):
    assert db.get_recent_track_ids(conn, limit=0) == []


def test_get_stats(conn):
    db.insert_track(conn, _sample_track(filepath="/a.mp3", genre="Techno", mood_happy=0.5))
    db.insert_track(conn, _sample_track(filepath="/b.mp3"))

    stats = db.get_stats(conn)
    assert stats["total_tracks"] == 2
    assert stats["with_genre"] == 1
    assert stats["with_mood"] == 1
    assert stats["with_bpm"] == 2


def test_musicbrainz_cache_roundtrip(conn):
    assert db.get_cached_lookup(conn, "artist|title") is None
    db.set_cached_lookup(conn, "artist|title", "some-mbid", "techno,house")
    cached = db.get_cached_lookup(conn, "artist|title")
    assert cached["mbid"] == "some-mbid"
    assert cached["genre_tags"] == "techno,house"


def test_normalize_energy_min_max_scaling(conn):
    db.insert_track(conn, _sample_track(filepath="/a.mp3", energy_raw=0.0))
    db.insert_track(conn, _sample_track(filepath="/b.mp3", energy_raw=0.5))
    db.insert_track(conn, _sample_track(filepath="/c.mp3", energy_raw=1.0))

    updated = db.normalize_energy(conn)
    assert updated == 3

    tracks = {t["filepath"]: t["energy"] for t in db.get_all_tracks(conn)}
    assert tracks["/a.mp3"] == pytest.approx(0.0)
    assert tracks["/b.mp3"] == pytest.approx(0.5)
    assert tracks["/c.mp3"] == pytest.approx(1.0)


def test_normalize_energy_single_value_uses_neutral_midpoint(conn):
    db.insert_track(conn, _sample_track(filepath="/a.mp3", energy_raw=0.42))
    db.insert_track(conn, _sample_track(filepath="/b.mp3", energy_raw=0.42))

    db.normalize_energy(conn)
    for track in db.get_all_tracks(conn):
        assert track["energy"] == pytest.approx(0.5)


def test_normalize_energy_skips_tracks_without_raw_value(conn):
    db.insert_track(conn, _sample_track(filepath="/a.mp3", energy_raw=0.0))
    db.insert_track(conn, _sample_track(filepath="/b.mp3", energy_raw=1.0))
    db.insert_track(conn, _sample_track(filepath="/c.mp3", energy=None))  # geen energy_raw

    db.normalize_energy(conn)
    tracks = {t["filepath"]: t["energy"] for t in db.get_all_tracks(conn)}
    assert tracks["/c.mp3"] is None


def test_normalize_energy_empty_db_returns_zero(conn):
    assert db.normalize_energy(conn) == 0


def test_connect_creates_parent_dir(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "dj.db"
    connection = db.connect(db_path)
    assert db_path.exists()
    connection.close()
