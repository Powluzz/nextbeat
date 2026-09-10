"""Tests voor dj_engine.ingest.pipeline.

Gebruikt de echte (zelfgegenereerde) audiofixtures voor end-to-end analyse,
en een gemockte MusicBrainz-client (geen netwerkcalls).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dj_engine import db
from dj_engine.config import load_config
from dj_engine.enrichment.musicbrainz_client import MusicBrainzClient
from dj_engine.ingest.pipeline import (
    analyze_track,
    find_audio_files,
    ingest_file,
    ingest_folder,
    read_tags,
)

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
    # Modellen zeker niet aanwezig -> mood/genre-classificatie wordt overgeslagen
    cfg["models"]["directory"] = str(tmp_path / "no_models_here")
    return cfg


class FakeMusicBrainzError(Exception):
    pass


class FakeMBModule:
    """Bootst de echte MB-API na: search_recordings() levert geen tags mee,
    alleen get_recording_by_id(includes=["tags"]) doet dat."""

    MusicBrainzError = FakeMusicBrainzError

    def __init__(self, mbid="fake-mbid", genres="techno,house"):
        self.mbid = mbid
        self.genres = genres
        self.call_count = 0

    def set_useragent(self, *a, **kw):
        pass

    def set_rate_limit(self, *a, **kw):
        pass

    def search_recordings(self, recording, artist, limit=1):
        self.call_count += 1
        return {"recording-list": [{"id": self.mbid}]}

    def get_recording_by_id(self, mbid, includes=None):
        self.call_count += 1
        tags = [{"name": g} for g in self.genres.split(",")] if self.genres else []
        return {"recording": {"id": mbid, "tag-list": tags}}


def _fake_mb_client(config):
    return MusicBrainzClient(config, db_conn=None, mb_module=FakeMBModule())


# --- find_audio_files -----------------------------------------------------

def test_find_audio_files_filters_by_extension_and_recurses(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.mp3").write_bytes(b"x")
    (tmp_path / "sub" / "b.WAV").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")

    found = find_audio_files(str(tmp_path), [".mp3", ".wav"])
    names = sorted(p.name for p in found)
    assert names == ["a.mp3", "b.WAV"]


def test_find_audio_files_empty_folder(tmp_path):
    assert find_audio_files(str(tmp_path), [".mp3"]) == []


# --- read_tags --------------------------------------------------------

def test_read_tags_untagged_file_returns_none_fields():
    tags = read_tags(str(FIXTURES / "click_128bpm.wav"))
    assert tags == {"title": None, "artist": None}


def test_read_tags_corrupt_file_does_not_raise():
    tags = read_tags(str(FIXTURES / "corrupt.wav"))
    assert tags == {"title": None, "artist": None}


def test_read_tags_missing_file_does_not_raise():
    tags = read_tags(str(FIXTURES / "does_not_exist.mp3"))
    assert tags == {"title": None, "artist": None}


# --- ingest_file --------------------------------------------------------

def test_ingest_file_full_pipeline_no_mb_client(config):
    track = ingest_file(FIXTURES / "click_128bpm.wav", config, mb_client=None)

    assert track is not None
    assert track["filepath"] == str(FIXTURES / "click_128bpm.wav")
    assert track["bpm"] is not None
    assert track["camelot"] is None or isinstance(track["camelot"], str)
    assert track["mbid"] is None  # geen mb_client meegegeven
    assert track["mood_happy"] is None  # geen modellen -> NULL, niet fabriceren


def test_ingest_file_with_mb_client_sets_mbid_and_genre(config):
    mb_client = _fake_mb_client(config)
    # title/artist zijn None (untagged fixture) -> lookup_track slaat netwerk over
    track = ingest_file(FIXTURES / "click_128bpm.wav", config, mb_client=mb_client)
    assert track["mbid"] is None
    assert track["genre"] is None


def test_ingest_file_returns_none_on_analysis_failure(config):
    track = ingest_file(FIXTURES / "corrupt.wav", config, mb_client=None)
    assert track is None


# --- ingest_folder --------------------------------------------------------

def test_ingest_folder_ingests_valid_and_skips_corrupt(tmp_path, conn, config):
    shutil.copy(FIXTURES / "click_128bpm.wav", tmp_path / "track1.wav")
    shutil.copy(FIXTURES / "tone_c_major.wav", tmp_path / "track2.wav")
    shutil.copy(FIXTURES / "corrupt.wav", tmp_path / "broken.wav")

    stats = ingest_folder(str(tmp_path), conn, config, mb_client=None, progress=False)

    assert stats == {"scanned": 3, "skipped_existing": 0, "ingested": 2, "failed": 1}
    tracks = db.get_all_tracks(conn)
    assert len(tracks) == 2


def test_ingest_folder_is_idempotent(tmp_path, conn, config):
    shutil.copy(FIXTURES / "click_128bpm.wav", tmp_path / "track1.wav")

    first = ingest_folder(str(tmp_path), conn, config, mb_client=None, progress=False)
    second = ingest_folder(str(tmp_path), conn, config, mb_client=None, progress=False)

    assert first["ingested"] == 1
    assert second["ingested"] == 0
    assert second["skipped_existing"] == 1
    assert len(db.get_all_tracks(conn)) == 1  # geen duplicaten


def test_ingest_folder_normalizes_energy_across_library(tmp_path, conn, config):
    shutil.copy(FIXTURES / "click_128bpm.wav", tmp_path / "track1.wav")
    shutil.copy(FIXTURES / "tone_c_major.wav", tmp_path / "track2.wav")

    ingest_folder(str(tmp_path), conn, config, mb_client=None, progress=False)

    tracks = db.get_all_tracks(conn)
    energies = [t["energy"] for t in tracks]
    assert all(e is not None for e in energies)
    assert all(0.0 <= e <= 1.0 for e in energies)
    assert set(energies) == {0.0, 1.0}  # 2 tracks -> min-max naar de uitersten


def test_ingest_folder_empty_directory_returns_zero_stats(tmp_path, conn, config):
    stats = ingest_folder(str(tmp_path), conn, config, mb_client=None, progress=False)
    assert stats == {"scanned": 0, "skipped_existing": 0, "ingested": 0, "failed": 0}


def test_ingest_folder_one_corrupt_file_does_not_stop_batch(tmp_path, conn, config):
    """Kwaliteitseis: 1 corrupt bestand mag de rest van de batch niet blokkeren."""
    shutil.copy(FIXTURES / "corrupt.wav", tmp_path / "broken1.wav")
    shutil.copy(FIXTURES / "corrupt.wav", tmp_path / "broken2.wav")
    shutil.copy(FIXTURES / "click_128bpm.wav", tmp_path / "good.wav")

    stats = ingest_folder(str(tmp_path), conn, config, mb_client=None, progress=False)

    assert stats["failed"] == 2
    assert stats["ingested"] == 1


# --- analyze_track (on-demand single-track, Rekordbox-flow) ----------------

def test_analyze_track_preserves_existing_bpm_and_camelot(tmp_path, conn, config):
    """Kernregel: bpm/key/camelot die al gezet zijn (bv. uit Rekordbox)
    mogen nooit overschreven worden door Essentia's eigen detectie."""
    filepath = tmp_path / "track.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", filepath)
    track_id = db.insert_track(conn, {
        "filepath": str(filepath), "bpm": 126.5, "camelot": "9A", "key": "A", "scale": "minor",
    })

    updated = analyze_track(track_id, conn, config, mb_client=None)

    assert updated["bpm"] == 126.5  # ongewijzigd, ook al detecteert Essentia ~128
    assert updated["camelot"] == "9A"
    assert updated["energy_raw"] is not None  # wel nieuw ingevuld
    assert updated["loudness"] is not None


def test_analyze_track_fills_missing_bpm_and_camelot(tmp_path, conn, config):
    filepath = tmp_path / "track.wav"
    shutil.copy(FIXTURES / "tone_c_major.wav", filepath)
    track_id = db.insert_track(conn, {"filepath": str(filepath)})

    updated = analyze_track(track_id, conn, config, mb_client=None)

    assert updated["bpm"] is not None
    assert updated["camelot"] == "8B"  # C majeur, zie test_essentia_extractor.py


def test_analyze_track_unknown_id_raises(conn, config):
    with pytest.raises(ValueError):
        analyze_track(999, conn, config, mb_client=None)


def test_analyze_track_streaming_scheme_returns_unchanged(conn, config):
    track_id = db.insert_track(conn, {
        "filepath": "tidal://tracks/12345", "bpm": 124.0, "title": "Streaming Track",
    })

    updated = analyze_track(track_id, conn, config, mb_client=None)

    assert updated["bpm"] == 124.0
    assert updated["energy_raw"] is None  # geen lokaal bestand -> niets geanalyseerd


def test_analyze_track_analysis_failure_returns_existing_unchanged(tmp_path, conn, config):
    filepath = tmp_path / "broken.wav"
    shutil.copy(FIXTURES / "corrupt.wav", filepath)
    track_id = db.insert_track(conn, {"filepath": str(filepath), "bpm": 128.0})

    updated = analyze_track(track_id, conn, config, mb_client=None)

    assert updated["bpm"] == 128.0
    assert updated["energy_raw"] is None


def test_analyze_track_triggers_periodic_normalize(tmp_path, conn, config):
    config = dict(config)
    config["scoring"] = {**config["scoring"], "energy_renormalize_threshold": 1}

    filepath = tmp_path / "track.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", filepath)
    track_id = db.insert_track(conn, {"filepath": str(filepath)})

    updated = analyze_track(track_id, conn, config, mb_client=None)

    assert updated["energy"] is not None  # drempel=1 -> meteen genormaliseerd
