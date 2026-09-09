"""Tests voor dj_engine.analysis.essentia_extractor.

Draait echte Essentia-analyse (geen mocks) op korte, zelfgegenereerde
(CC0/eigen bezit) audiofragmenten in tests/fixtures/. Zie
tests/fixtures/generate_fixtures.py voor hoe deze gemaakt zijn.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dj_engine.analysis.essentia_extractor import analyze_audio

pytest.importorskip("essentia", reason="essentia is een optionele dependency")

FIXTURES = Path(__file__).parent / "fixtures"


def test_analyze_click_track_detects_bpm_near_128():
    result = analyze_audio(str(FIXTURES / "click_128bpm.wav"))

    assert result["error"] is None
    assert result["bpm"] is not None
    # RhythmExtractor2013 kan af en toe een octaaffout maken (halve/dubbele
    # tempo); we accepteren 128 of de bekende harmonischen, met tolerantie.
    plausible_bpms = [64.0, 128.0, 256.0]
    assert any(abs(result["bpm"] - p) / p < 0.10 for p in plausible_bpms), (
        f"bpm={result['bpm']} niet in de buurt van {plausible_bpms}"
    )
    assert result["bpm_confidence"] is not None


def test_analyze_tone_detects_c_major_key():
    result = analyze_audio(str(FIXTURES / "tone_c_major.wav"))

    assert result["error"] is None
    assert result["key"] == "C"
    assert result["scale"] == "major"
    assert result["camelot"] == "8B"
    assert result["key_strength"] > 0.5


def test_analyze_returns_all_expected_fields():
    result = analyze_audio(str(FIXTURES / "click_128bpm.wav"))

    expected_keys = {
        "filepath", "bpm", "bpm_confidence", "key", "scale", "key_strength",
        "camelot", "loudness", "danceability", "energy_raw", "error",
    }
    assert set(result.keys()) == expected_keys
    assert result["loudness"] is not None
    assert result["danceability"] is not None
    assert result["energy_raw"] is not None
    assert result["energy_raw"] >= 0.0


def test_analyze_silence_does_not_crash_and_flags_low_confidence(caplog):
    with caplog.at_level("WARNING"):
        result = analyze_audio(str(FIXTURES / "silence.wav"))

    # Stilte crasht niet: Essentia levert (onbetrouwbare) waarden, geen error.
    assert result["error"] is None
    assert result["key_strength"] == 0.0
    assert result["energy_raw"] == 0.0
    assert "Lage" in caplog.text  # lage-confidence waarschuwing gelogd


def test_analyze_corrupt_file_returns_error_not_exception():
    result = analyze_audio(str(FIXTURES / "corrupt.wav"))

    assert result["error"] is not None
    assert result["bpm"] is None
    assert result["key"] is None
    assert result["camelot"] is None


def test_analyze_missing_file_returns_error_not_exception():
    result = analyze_audio(str(FIXTURES / "does_not_exist.wav"))

    assert result["error"] is not None
    assert "niet gevonden" in result["error"]
    assert result["bpm"] is None


def test_analyze_never_raises_on_batch_of_fixtures():
    """Smoke test: alle fixtures (incl. corrupte) achter elkaar analyseren
    mag nooit een exception opgooien — precies het gedrag dat de
    ingest-pipeline nodig heeft om door te gaan bij een kapot bestand."""
    for fixture in FIXTURES.glob("*.wav"):
        result = analyze_audio(str(fixture))
        assert isinstance(result, dict)
        assert "error" in result
