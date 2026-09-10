"""Tests voor dj_engine.recommend.refine.

Draait echte (spawned) workerprocessen tegen de zelfgegenereerde
audiofixtures — geen mocks, dit test net zo goed de multiprocessing-
orchestratie zelf. Modellen-map wijst naar een lege tmp-map zodat de
workers niet steeds opnieuw de echte (grote) TF-modellen hoeven te laden
— dat pad is al apart gedekt in test_mood_genre_models.py.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dj_engine import db
from dj_engine.config import load_config
from dj_engine.recommend.refine import refine_candidates

pytest.importorskip("essentia", reason="essentia is een optionele dependency")

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg["models"]["directory"] = str(tmp_path / "no_models_here")
    cfg["analysis_pool"] = {"workers": 2}
    return cfg


def test_refine_candidates_fills_missing_energy(tmp_path, conn, config):
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", f1)
    shutil.copy(FIXTURES / "tone_c_major.wav", f2)
    id1 = db.insert_track(conn, {"filepath": str(f1)})
    id2 = db.insert_track(conn, {"filepath": str(f2)})

    refined = refine_candidates(conn, config, [id1, id2])

    assert refined == 2
    t1 = db.get_track(conn, id1)
    t2 = db.get_track(conn, id2)
    assert t1["energy_raw"] is not None
    assert t2["energy_raw"] is not None
    assert t1["energy"] is not None  # meteen genormaliseerd
    assert t2["energy"] is not None


def test_refine_candidates_preserves_existing_bpm_camelot(tmp_path, conn, config):
    f = tmp_path / "a.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", f)
    track_id = db.insert_track(conn, {"filepath": str(f), "bpm": 126.5, "camelot": "9A"})

    refine_candidates(conn, config, [track_id])

    updated = db.get_track(conn, track_id)
    assert updated["bpm"] == 126.5
    assert updated["camelot"] == "9A"


def test_refine_candidates_skips_already_analyzed(tmp_path, conn, config):
    f = tmp_path / "a.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", f)
    track_id = db.insert_track(conn, {
        "filepath": str(f), "energy_raw": 0.5, "mood_happy": 0.7,
    })

    refined = refine_candidates(conn, config, [track_id])
    assert refined == 0


def test_refine_candidates_skips_streaming_tracks(conn, config):
    track_id = db.insert_track(conn, {"filepath": "tidal://tracks/999"})
    refined = refine_candidates(conn, config, [track_id])
    assert refined == 0


def test_refine_candidates_skips_unknown_ids(conn, config):
    assert refine_candidates(conn, config, [999999]) == 0


def test_refine_candidates_empty_list_returns_zero(conn, config):
    assert refine_candidates(conn, config, []) == 0


def test_refine_candidates_handles_corrupt_file_without_crashing(tmp_path, conn, config):
    f = tmp_path / "broken.wav"
    shutil.copy(FIXTURES / "corrupt.wav", f)
    track_id = db.insert_track(conn, {"filepath": str(f)})

    refined = refine_candidates(conn, config, [track_id])

    assert refined == 0
    assert db.get_track(conn, track_id)["energy_raw"] is None


def test_refine_candidates_mixed_batch_partial_success(tmp_path, conn, config):
    """Eén corrupt bestand in de batch mag de andere kandidaten niet blokkeren."""
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", good)
    shutil.copy(FIXTURES / "corrupt.wav", bad)
    good_id = db.insert_track(conn, {"filepath": str(good)})
    bad_id = db.insert_track(conn, {"filepath": str(bad)})

    refined = refine_candidates(conn, config, [good_id, bad_id])

    assert refined == 1
    assert db.get_track(conn, good_id)["energy_raw"] is not None
    assert db.get_track(conn, bad_id)["energy_raw"] is None
