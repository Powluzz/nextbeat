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
