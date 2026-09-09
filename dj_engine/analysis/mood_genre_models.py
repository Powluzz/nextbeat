"""Optionele mood/genre-classificatie via pretrained Essentia TensorFlow-modellen.

Gebruikt de Discogs-EffNet embedding + downstream classifiers-aanpak van
Essentia (zie https://essentia.upf.edu/models.html) voor mood_happy,
mood_sad, mood_aggressive, mood_relaxed, mood_party en genre_discogs400.

Vereist het `essentia-tensorflow`-pakket (niet hetzelfde als het basis
`essentia`-pakket) én de modelbestanden in `models/` — zie
`models/download_models.sh`. In deze projectomgeving is `essentia-tensorflow`
niet beschikbaar als wheel voor dit platform/deze Python-versie, dus dit pad
is hier niet end-to-end getest tegen echte modellen; het is geschreven
volgens Essentia's gedocumenteerde inference-patroon en degradeert altijd
veilig (zie `models_available()` hieronder).

**Belangrijk (kwaliteitseis):** als de modelbestanden niet aanwezig zijn,
wordt classificatie overgeslagen en blijven de betreffende
databasekolommen NULL. Er wordt nooit data verzonnen.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-geladen modellen, per (embedding_model_path, mood/genre_model_path)
# zodat ze maar één keer per proces van schijf komen, niet per track.
_MODEL_CACHE: dict[str, Any] = {}

_EMPTY_RESULT: dict[str, Any] = {
    "mood_happy": None,
    "mood_sad": None,
    "mood_aggressive": None,
    "mood_relaxed": None,
    "mood_party": None,
    "genre": None,
}

# Discogs400-genrelabels worden uit het meegeleverde .json-bestand gelezen
# (niet hardcoded) omdat de exacte labelset aan het modelbestand gebonden is.


def _required_model_paths(config: dict[str, Any]) -> list[Path]:
    models_cfg = config["models"]
    directory = Path(models_cfg["directory"])
    paths = [directory / models_cfg["embedding_model"]]
    paths += [directory / fname for fname in models_cfg["mood_models"].values()]
    paths.append(directory / models_cfg["genre_model"])
    paths.append(directory / models_cfg["genre_labels"])
    return paths


def models_available(config: dict[str, Any]) -> bool:
    """True als alle benodigde modelbestanden aanwezig zijn op schijf."""
    return all(p.exists() for p in _required_model_paths(config))


def classify_mood_genre(filepath: str, config: dict[str, Any]) -> dict[str, Any]:
    """Classificeer mood + genre voor een audiobestand.

    Args:
        filepath: pad naar het audiobestand (wordt apart ingeladen op de
            sample rate die het embedding-model verwacht — 16kHz voor
            Discogs-EffNet — los van de 44.1kHz die essentia_extractor
            gebruikt voor BPM/key/loudness).
        config: app-config (zie config.yaml -> models).

    Returns:
        dict met mood_happy/mood_sad/mood_aggressive/mood_relaxed/
        mood_party/genre. Alle waarden None als modellen ontbreken of
        classificatie faalt — nooit gefabriceerd, altijd gelogd.
    """
    if not models_available(config):
        logger.warning(
            "Mood/genre-modellen niet gevonden in '%s' — mood/genre-kolommen "
            "blijven NULL voor %s. Draai models/download_models.sh om ze op te halen.",
            config["models"]["directory"], filepath,
        )
        return dict(_EMPTY_RESULT)

    try:
        import essentia.standard as es
    except ImportError as exc:
        logger.warning(
            "essentia (met TensorFlow-support) niet beschikbaar voor "
            "mood/genre-classificatie van %s: %s", filepath, exc,
        )
        return dict(_EMPTY_RESULT)

    if not hasattr(es, "TensorflowPredictEffnetDiscogs"):
        logger.warning(
            "Dit essentia-pakket is gebouwd zonder TensorFlow-support "
            "(installeer 'essentia-tensorflow' i.p.v. 'essentia') — "
            "mood/genre-kolommen blijven NULL voor %s.", filepath,
        )
        return dict(_EMPTY_RESULT)

    try:
        return _run_inference(es, filepath, config)
    except Exception as exc:
        logger.error("Mood/genre-classificatie mislukt voor %s: %s", filepath, exc)
        return dict(_EMPTY_RESULT)


def _run_inference(es_module, filepath: str, config: dict[str, Any]) -> dict[str, Any]:
    models_cfg = config["models"]
    directory = Path(models_cfg["directory"])

    # Discogs-EffNet verwacht 16kHz mono audio.
    audio = es_module.MonoLoader(filename=str(filepath), sampleRate=16000, resampleQuality=4)()

    embedding_model_path = str(directory / models_cfg["embedding_model"])
    embeddings = _get_or_load(
        es_module, "embedding", embedding_model_path,
        lambda: es_module.TensorflowPredictEffnetDiscogs(
            graphFilename=embedding_model_path, output="PartitionedCall:1"
        ),
    )(audio)

    result: dict[str, Any] = dict(_EMPTY_RESULT)

    for mood_name, model_fname in models_cfg["mood_models"].items():
        model_path = str(directory / model_fname)
        predictor = _get_or_load(
            es_module, f"mood::{mood_name}", model_path,
            lambda p=model_path: es_module.TensorflowPredict2D(graphFilename=p),
        )
        predictions = predictor(embeddings)
        # Binaire mood-classifiers geven doorgaans [P(afwezig), P(aanwezig)].
        result[mood_name] = float(predictions.mean(axis=0)[-1])

    genre_model_path = str(directory / models_cfg["genre_model"])
    genre_predictor = _get_or_load(
        es_module, "genre", genre_model_path,
        lambda: es_module.TensorflowPredict2D(graphFilename=genre_model_path),
    )
    genre_predictions = genre_predictor(embeddings).mean(axis=0)

    labels = _load_genre_labels(directory / models_cfg["genre_labels"])
    if labels:
        top_index = int(genre_predictions.argmax())
        result["genre"] = labels[top_index] if top_index < len(labels) else None

    return result


def _get_or_load(es_module, cache_key: str, model_path: str, factory):
    full_key = f"{cache_key}::{model_path}"
    if full_key not in _MODEL_CACHE:
        _MODEL_CACHE[full_key] = factory()
    return _MODEL_CACHE[full_key]


def _load_genre_labels(labels_path: Path) -> list[str] | None:
    import json

    if not labels_path.exists():
        return None
    try:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        # Essentia-labelbestanden zijn ofwel een platte lijst, ofwel
        # {"classes": [...]}
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "classes" in data:
            return data["classes"]
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Kon genre-labels niet lezen uit %s: %s", labels_path, exc)
        return None
