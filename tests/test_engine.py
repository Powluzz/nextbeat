"""Tests voor dj_engine.recommend.engine.suggest_next()."""
from __future__ import annotations

import pytest

from dj_engine import db
from dj_engine.config import load_config
from dj_engine.recommend.engine import suggest_next


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def config():
    return load_config()


def _insert(conn, **overrides):
    track = {
        "filepath": overrides.pop("filepath", f"/music/{overrides.get('title', 'x')}.mp3"),
        "title": "Untitled",
        "artist": "Unknown",
        "bpm": 128.0,
        "camelot": "8A",
        "energy": 0.5,
    }
    track.update(overrides)
    return db.insert_track(conn, track)


def test_suggest_next_ranks_best_key_bpm_match_first(conn, config):
    current_id = _insert(conn, title="current", bpm=128.0, camelot="8A", energy=0.5)
    perfect_id = _insert(conn, title="perfect", bpm=128.0, camelot="8A", energy=0.5)
    poor_id = _insert(conn, title="poor", bpm=180.0, camelot="3B", energy=0.5)

    results = suggest_next(conn, current_id, "hold", config, top_n=10)

    ids_in_order = [r["track"]["id"] for r in results]
    assert ids_in_order[0] == perfect_id
    assert ids_in_order[-1] == poor_id


def test_suggest_next_excludes_current_track(conn, config):
    current_id = _insert(conn, title="current")
    other_id = _insert(conn, title="other")

    results = suggest_next(conn, current_id, "hold", config, top_n=10)

    ids = [r["track"]["id"] for r in results]
    assert current_id not in ids
    assert other_id in ids


def test_suggest_next_respects_explicit_exclude_list(conn, config):
    current_id = _insert(conn, title="current")
    excluded_id = _insert(conn, title="excluded")
    other_id = _insert(conn, title="other")

    results = suggest_next(
        conn, current_id, "hold", config, top_n=10, exclude_recently_played=[excluded_id]
    )

    ids = [r["track"]["id"] for r in results]
    assert excluded_id not in ids
    assert other_id in ids


def test_suggest_next_auto_excludes_recent_transitions(conn, config):
    current_id = _insert(conn, title="current")
    recent_id = _insert(conn, title="recent")
    fresh_id = _insert(conn, title="fresh")

    db.log_transition(conn, current_id, recent_id, "hold")

    # exclude_recently_played=None (default) -> gebruikt transitions-tabel
    results = suggest_next(conn, current_id, "hold", config, top_n=10)

    ids = [r["track"]["id"] for r in results]
    assert recent_id not in ids
    assert fresh_id in ids


def test_suggest_next_empty_exclude_list_disables_auto_exclusion(conn, config):
    current_id = _insert(conn, title="current")
    recent_id = _insert(conn, title="recent")
    db.log_transition(conn, current_id, recent_id, "hold")

    results = suggest_next(conn, current_id, "hold", config, top_n=10, exclude_recently_played=[])

    ids = [r["track"]["id"] for r in results]
    assert recent_id in ids  # expliciet lege lijst -> geen auto-uitsluiting


def test_suggest_next_respects_top_n(conn, config):
    current_id = _insert(conn, title="current")
    for i in range(5):
        _insert(conn, title=f"track{i}")

    results = suggest_next(conn, current_id, "hold", config, top_n=3)
    assert len(results) == 3


def test_suggest_next_returns_score_breakdown_per_dimension(conn, config):
    current_id = _insert(conn, title="current")
    _insert(conn, title="other")

    results = suggest_next(conn, current_id, "hold", config, top_n=10)

    assert len(results) == 1
    result = results[0]
    assert set(result.keys()) == {"track", "total_score", "breakdown"}
    assert set(result["breakdown"].keys()) == {"key", "bpm", "energy", "mood"}
    assert isinstance(result["total_score"], float)


def test_suggest_next_unknown_direction_raises(conn, config):
    current_id = _insert(conn, title="current")
    with pytest.raises(ValueError):
        suggest_next(conn, current_id, "sideways", config)


def test_suggest_next_unknown_track_id_raises(conn, config):
    with pytest.raises(ValueError):
        suggest_next(conn, 999, "hold", config)


def test_suggest_next_build_direction_prefers_higher_energy(conn, config):
    current_id = _insert(conn, title="current", bpm=128, camelot="8A", energy=0.4)
    higher_id = _insert(conn, title="higher", bpm=128, camelot="8A", energy=0.65)
    lower_id = _insert(conn, title="lower", bpm=128, camelot="8A", energy=0.2)

    results = suggest_next(conn, current_id, "build", config, top_n=10)
    ids_in_order = [r["track"]["id"] for r in results]
    assert ids_in_order.index(higher_id) < ids_in_order.index(lower_id)


def test_suggest_next_ease_direction_prefers_lower_energy(conn, config):
    current_id = _insert(conn, title="current", bpm=128, camelot="8A", energy=0.6)
    higher_id = _insert(conn, title="higher", bpm=128, camelot="8A", energy=0.85)
    lower_id = _insert(conn, title="lower", bpm=128, camelot="8A", energy=0.35)

    results = suggest_next(conn, current_id, "ease", config, top_n=10)
    ids_in_order = [r["track"]["id"] for r in results]
    assert ids_in_order.index(lower_id) < ids_in_order.index(higher_id)


def test_suggest_next_no_other_tracks_returns_empty(conn, config):
    current_id = _insert(conn, title="current")
    results = suggest_next(conn, current_id, "hold", config, top_n=10)
    assert results == []
