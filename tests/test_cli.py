"""Tests voor dj_engine.cli via click.testing.CliRunner.

Logging-configuratie wordt genoopt (zie `no_configure_logging` hieronder):
`main()` roept anders bij elke invoke() logging.basicConfig(force=True) aan,
wat pytest's eigen log-capturing in andere testmodules zou verstoren.
Dat gedrag van configure_logging() zelf staat apart getest in
test_logging_config.py.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from dj_engine import db
from dj_engine.cli import main
from dj_engine.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_configure_logging(monkeypatch):
    monkeypatch.setattr("dj_engine.cli.configure_logging", lambda config: None)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def config_path(tmp_path):
    """Schrijft een config.yaml die naar een tmp-database wijst, met een
    lege (niet-bestaande) modellenmap zodat mood/genre altijd NULL blijft."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "database": {"path": str(tmp_path / "dj.db")},
                "models": {"directory": str(tmp_path / "no_models")},
                "logging": {"file": str(tmp_path / "dj.log")},
                "analysis_pool": {"workers": 2},
            }
        )
    )
    return str(cfg_file)


@pytest.fixture
def music_dir(tmp_path):
    d = tmp_path / "music"
    d.mkdir()
    shutil.copy(FIXTURES / "click_128bpm.wav", d / "track1.wav")
    shutil.copy(FIXTURES / "tone_c_major.wav", d / "track2.wav")
    shutil.copy(FIXTURES / "corrupt.wav", d / "broken.wav")
    return str(d)


def _invoke(runner, config_path, args):
    return runner.invoke(main, ["--config", config_path, *args])


# --- ingest --------------------------------------------------------

def test_cli_ingest_reports_stats(runner, config_path, music_dir):
    result = _invoke(runner, config_path, ["ingest", music_dir, "--no-musicbrainz", "--no-progress"])

    assert result.exit_code == 0, result.output
    assert "2 nieuw toegevoegd" in result.output
    assert "1 mislukt" in result.output


def test_cli_ingest_is_idempotent(runner, config_path, music_dir):
    _invoke(runner, config_path, ["ingest", music_dir, "--no-musicbrainz", "--no-progress"])
    result = _invoke(runner, config_path, ["ingest", music_dir, "--no-musicbrainz", "--no-progress"])

    assert result.exit_code == 0
    assert "0 nieuw toegevoegd" in result.output
    assert "2 overgeslagen" in result.output


def test_cli_ingest_nonexistent_folder_errors(runner, config_path):
    result = _invoke(runner, config_path, ["ingest", "/does/not/exist", "--no-musicbrainz"])
    assert result.exit_code != 0


# --- stats --------------------------------------------------------

def test_cli_stats_empty_database(runner, config_path):
    result = _invoke(runner, config_path, ["stats"])
    assert result.exit_code == 0
    assert "Totaal aantal tracks: 0" in result.output


def test_cli_stats_after_ingest(runner, config_path, music_dir):
    _invoke(runner, config_path, ["ingest", music_dir, "--no-musicbrainz", "--no-progress"])
    result = _invoke(runner, config_path, ["stats"])

    assert result.exit_code == 0
    assert "Totaal aantal tracks: 2" in result.output
    assert "Met mood-data" in result.output


# --- search --------------------------------------------------------

def test_cli_search_by_bpm_range(runner, config_path, music_dir):
    _invoke(runner, config_path, ["ingest", music_dir, "--no-musicbrainz", "--no-progress"])
    result = _invoke(runner, config_path, ["search", "--bpm-range", "100-200"])

    assert result.exit_code == 0
    assert "track(s) gevonden" in result.output


def test_cli_search_invalid_bpm_range_errors(runner, config_path):
    result = _invoke(runner, config_path, ["search", "--bpm-range", "not-a-range"])
    assert result.exit_code != 0


def test_cli_search_no_results(runner, config_path):
    result = _invoke(runner, config_path, ["search", "--artist", "Nobody"])
    assert result.exit_code == 0
    assert "Geen tracks gevonden" in result.output


# --- suggest --------------------------------------------------------

def test_cli_suggest_after_ingest(runner, config_path, music_dir):
    _invoke(runner, config_path, ["ingest", music_dir, "--no-musicbrainz", "--no-progress"])
    result = _invoke(runner, config_path, ["suggest", "1", "--direction", "build", "--top", "5"])

    assert result.exit_code == 0, result.output
    assert "Richting: build" in result.output
    assert "totaal=" in result.output


def test_cli_suggest_unknown_track_id_errors(runner, config_path):
    result = _invoke(runner, config_path, ["suggest", "999"])
    assert result.exit_code != 0
    assert "niet gevonden" in result.output


def test_cli_suggest_invalid_direction_errors(runner, config_path):
    result = _invoke(runner, config_path, ["suggest", "1", "--direction", "sideways"])
    assert result.exit_code != 0


def test_cli_suggest_shows_score_breakdown(runner, config_path):
    """Vereiste: de gebruiker moet zien WAAROM een track wordt voorgesteld."""
    cfg = load_config(config_path)
    conn = db.connect(cfg["database"]["path"])
    db.insert_track(conn, {"filepath": "/a.mp3", "title": "A", "artist": "X", "bpm": 128, "camelot": "8A", "energy": 0.5})
    db.insert_track(conn, {"filepath": "/b.mp3", "title": "B", "artist": "Y", "bpm": 129, "camelot": "9A", "energy": 0.6})
    conn.close()

    result = _invoke(runner, config_path, ["suggest", "1", "--direction", "hold"])

    assert result.exit_code == 0, result.output
    assert "key=" in result.output and "bpm=" in result.output
    assert "energy=" in result.output and "mood=" in result.output


def test_cli_suggest_no_refine_skips_refinement(runner, config_path):
    cfg = load_config(config_path)
    conn = db.connect(cfg["database"]["path"])
    db.insert_track(conn, {"filepath": "/a.mp3", "bpm": 128, "camelot": "8A"})
    db.insert_track(conn, {"filepath": "/b.mp3", "bpm": 129, "camelot": "9A"})
    conn.close()

    result = _invoke(runner, config_path, ["suggest", "1", "--no-refine"])

    assert result.exit_code == 0, result.output
    assert "Snel" in result.output
    assert "Verfijnen" not in result.output
    assert "Ververst" not in result.output


def test_cli_suggest_shows_fast_then_refresh_on_real_files(runner, config_path, tmp_path):
    """Kernvereiste: eerst de snelle score tonen, dan verfijnen met echte
    energie/mood-data (zie ingest_rekordbox-workflow)."""
    music_dir = tmp_path / "refine_music"
    music_dir.mkdir()
    f1 = music_dir / "current.wav"
    f2 = music_dir / "candidate.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", f1)
    shutil.copy(FIXTURES / "tone_c_major.wav", f2)

    cfg = load_config(config_path)
    conn = db.connect(cfg["database"]["path"])
    # Simuleert Rekordbox-import: bpm/camelot bekend, energy/mood nog niet.
    db.insert_track(conn, {"filepath": str(f1), "bpm": 128.0, "camelot": "8A"})
    db.insert_track(conn, {"filepath": str(f2), "bpm": 120.0, "camelot": "8B"})
    conn.close()

    result = _invoke(runner, config_path, ["suggest", "1", "--direction", "hold"])

    assert result.exit_code == 0, result.output
    assert "-- Snel (bpm/key/genre) --" in result.output
    assert "Verfijnen met energie/mood-analyse" in result.output
    assert "aangevuld" in result.output
    assert "-- Ververst" in result.output


# --- import-rekordbox --------------------------------------------------

REKORDBOX_FIXTURE = FIXTURES / "rekordbox_sample.xml"


def test_cli_import_rekordbox_reports_counts(runner, config_path):
    result = _invoke(runner, config_path, ["import-rekordbox", str(REKORDBOX_FIXTURE)])

    assert result.exit_code == 0, result.output
    assert "6 tracks verwerkt" in result.output
    assert "1 streaming-tracks" in result.output
    assert "path_mapping" in result.output  # waarschuwing: nog niet ingevuld


def test_cli_import_rekordbox_is_idempotent(runner, config_path):
    _invoke(runner, config_path, ["import-rekordbox", str(REKORDBOX_FIXTURE)])
    cfg = load_config(config_path)
    conn = db.connect(cfg["database"]["path"])
    count_before = len(db.get_all_tracks(conn))
    conn.close()

    _invoke(runner, config_path, ["import-rekordbox", str(REKORDBOX_FIXTURE)])
    conn = db.connect(cfg["database"]["path"])
    count_after = len(db.get_all_tracks(conn))
    conn.close()

    assert count_before == count_after == 6


def test_cli_import_rekordbox_fills_bpm_key_camelot(runner, config_path):
    _invoke(runner, config_path, ["import-rekordbox", str(REKORDBOX_FIXTURE)])
    cfg = load_config(config_path)
    conn = db.connect(cfg["database"]["path"])
    track = db.search_tracks(conn, artist="Robin Thicke")[0]
    conn.close()

    assert track["bpm"] == 120.0
    assert track["camelot"] == "10B"
    assert track["energy"] is None  # nog niet door de engine


def test_cli_import_rekordbox_missing_file_errors(runner, config_path):
    result = _invoke(runner, config_path, ["import-rekordbox", "/does/not/exist.xml"])
    assert result.exit_code != 0


# --- analyze --------------------------------------------------------

def test_cli_analyze_fills_energy_preserves_bpm(runner, config_path, tmp_path):
    f = tmp_path / "track.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", f)
    cfg = load_config(config_path)
    conn = db.connect(cfg["database"]["path"])
    track_id = db.insert_track(conn, {"filepath": str(f), "bpm": 126.5, "camelot": "9A"})
    conn.close()

    result = _invoke(runner, config_path, ["analyze", str(track_id), "--no-musicbrainz"])

    assert result.exit_code == 0, result.output
    assert "geanalyseerd" in result.output
    conn = db.connect(cfg["database"]["path"])
    track = db.get_track(conn, track_id)
    conn.close()
    assert track["bpm"] == 126.5  # onaangeroerd
    assert track["energy_raw"] is not None


def test_cli_analyze_unknown_id_errors(runner, config_path):
    result = _invoke(runner, config_path, ["analyze", "999", "--no-musicbrainz"])
    assert result.exit_code != 0
    assert "Fout" in result.output


# --- normalize-energy --------------------------------------------------

def test_cli_normalize_energy(runner, config_path):
    cfg = load_config(config_path)
    conn = db.connect(cfg["database"]["path"])
    db.insert_track(conn, {"filepath": "/a.mp3", "energy_raw": 0.0})
    db.insert_track(conn, {"filepath": "/b.mp3", "energy_raw": 1.0})
    conn.close()

    result = _invoke(runner, config_path, ["normalize-energy"])

    assert result.exit_code == 0, result.output
    assert "2 tracks genormaliseerd" in result.output


# --- set-api-key --------------------------------------------------------

def test_cli_set_api_key_writes_env_file(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["set-api-key", "sk-ant-test123"])

    assert result.exit_code == 0, result.output
    env_content = (tmp_path / ".env").read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-test123" in env_content


def test_cli_set_api_key_replaces_existing_value(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=old-value\nOTHER_VAR=keep-me\n")

    runner.invoke(main, ["set-api-key", "new-value"])

    env_content = (tmp_path / ".env").read_text()
    assert "ANTHROPIC_API_KEY=new-value" in env_content
    assert "old-value" not in env_content
    assert "OTHER_VAR=keep-me" in env_content


def test_cli_set_api_key_custom_var_name(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["set-api-key", "abc", "--var-name", "MY_KEY"])

    assert result.exit_code == 0
    assert "MY_KEY=abc" in (tmp_path / ".env").read_text()
