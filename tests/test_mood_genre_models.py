"""Tests voor dj_engine.analysis.mood_genre_models.

Twee delen:
1. Het 'modellen ontbreken'-pad (kwaliteitseis: nooit fabriceren) — draait
   altijd, met tmp_path-fixtures, geen echte modellen nodig.
2. Echte inference tegen de gedownloade modellen in models/ (zie
   models/download_models.sh) — wordt overgeslagen als die er niet zijn
   of als 'essentia-tensorflow' niet geïnstalleerd is (dan is er alleen
   het basis 'essentia'-pakket, zonder TensorFlow-support).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dj_engine.analysis.mood_genre_models import classify_mood_genre, models_available
from dj_engine.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"
REAL_CONFIG = load_config()

pytest.importorskip("essentia", reason="essentia is een optionele dependency")
import essentia.standard as _es  # noqa: E402

_has_tf_support = hasattr(_es, "TensorflowPredictEffnetDiscogs")
_real_models_present = models_available(REAL_CONFIG)

requires_real_models = pytest.mark.skipif(
    not (_has_tf_support and _real_models_present),
    reason=(
        "essentia-tensorflow + gedownloade modellen nodig "
        "(bash models/download_models.sh) voor echte inference-tests"
    ),
)


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


# --- Echte inference tegen gedownloade modellen --------------------------

@requires_real_models
def test_classify_real_models_returns_plausible_values():
    result = classify_mood_genre(str(FIXTURES / "click_128bpm.wav"), REAL_CONFIG)

    mood_keys = ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party"]
    for key in mood_keys:
        assert result[key] is not None, f"{key} zou gevuld moeten zijn met echte modellen"
        assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} buiten [0,1]"

    assert result["genre"] is not None
    assert isinstance(result["genre"], str) and len(result["genre"]) > 0


@requires_real_models
def test_classify_real_models_no_error_logged(caplog):
    with caplog.at_level("WARNING"):
        result = classify_mood_genre(str(FIXTURES / "tone_c_major.wav"), REAL_CONFIG)

    assert all(v is not None for v in result.values())
    assert "mislukt" not in caplog.text
    assert "niet gevonden" not in caplog.text


@requires_real_models
def test_classify_real_models_is_deterministic():
    """Zelfde input -> zelfde output (geen willekeur/fabricatie)."""
    first = classify_mood_genre(str(FIXTURES / "click_128bpm.wav"), REAL_CONFIG)
    second = classify_mood_genre(str(FIXTURES / "click_128bpm.wav"), REAL_CONFIG)
    assert first == second


@requires_real_models
def test_classify_real_models_genre_label_matches_known_taxonomy():
    """Genre-label moet uit de daadwerkelijke Discogs400-labelset komen
    (nooit een verzonnen string) — formaat 'Hoofdgenre---Subgenre'."""
    import json

    labels_path = Path(REAL_CONFIG["models"]["directory"]) / REAL_CONFIG["models"]["genre_labels"]
    known_labels = set(json.loads(labels_path.read_text())["classes"])

    result = classify_mood_genre(str(FIXTURES / "click_128bpm.wav"), REAL_CONFIG)
    assert result["genre"] in known_labels
