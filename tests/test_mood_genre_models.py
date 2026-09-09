"""Tests voor dj_engine.analysis.mood_genre_models.

Focus op de kwaliteitseis 'geen gefabriceerde data': als de modelbestanden
ontbreken (het geval in deze projectomgeving — essentia-tensorflow is hier
niet geïnstalleerd), moeten alle mood/genre-velden None zijn en moet dat
duidelijk gelogd worden.
"""
from __future__ import annotations

from pathlib import Path

from dj_engine.analysis.mood_genre_models import classify_mood_genre, models_available


def _config(tmp_path: Path) -> dict:
    return {
        "models": {
            "directory": str(tmp_path),
            "embedding_model": "discogs-effnet-bs64-1.pb",
            "mood_models": {
                "mood_happy": "mood_happy-discogs-effnet-1.pb",
                "mood_sad": "mood_sad-discogs-effnet-1.pb",
                "mood_aggressive": "mood_aggressive-discogs-effnet-1.pb",
                "mood_relaxed": "mood_relaxed-discogs-effnet-1.pb",
                "mood_party": "mood_party-discogs-effnet-1.pb",
            },
            "genre_model": "genre_discogs400-discogs-effnet-1.pb",
            "genre_labels": "genre_discogs400-discogs-effnet-1.json",
        }
    }


def test_models_available_false_when_directory_empty(tmp_path):
    assert models_available(_config(tmp_path)) is False


def test_models_available_true_when_all_files_present(tmp_path):
    config = _config(tmp_path)
    for fname in [
        config["models"]["embedding_model"],
        *config["models"]["mood_models"].values(),
        config["models"]["genre_model"],
        config["models"]["genre_labels"],
    ]:
        (tmp_path / fname).write_bytes(b"fake-model-bytes")
    assert models_available(config) is True


def test_models_available_false_when_one_file_missing(tmp_path):
    config = _config(tmp_path)
    # Alles behalve het genre-model aanmaken
    (tmp_path / config["models"]["embedding_model"]).write_bytes(b"x")
    for fname in config["models"]["mood_models"].values():
        (tmp_path / fname).write_bytes(b"x")
    assert models_available(config) is False


def test_classify_returns_all_none_when_models_missing(tmp_path, caplog):
    config = _config(tmp_path)
    with caplog.at_level("WARNING"):
        result = classify_mood_genre("/some/track.mp3", config)

    assert result == {
        "mood_happy": None,
        "mood_sad": None,
        "mood_aggressive": None,
        "mood_relaxed": None,
        "mood_party": None,
        "genre": None,
    }
    assert "download_models.sh" in caplog.text


def test_classify_never_raises_when_models_missing(tmp_path):
    config = _config(tmp_path)
    # Mag geen exception geven, ook niet met een niet-bestaand audiobestand —
    # de models_available()-check moet al vroeg terugkeren.
    result = classify_mood_genre("/nonexistent/file.mp3", config)
    assert all(v is None for v in result.values())
