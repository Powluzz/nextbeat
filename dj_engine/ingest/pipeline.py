"""Ingest-pipeline: scan map -> tags lezen -> audio-analyse ->
mood/genre-classificatie (optioneel) -> MusicBrainz-verrijking -> database.

Kwaliteitseisen uit BUILD_SPEC.md die deze module waarborgt:
- Idempotent: bestanden die al in de database staan (op filepath) worden
  overgeslagen, nooit opnieuw geanalyseerd of gedupliceerd.
- Foutafhandeling per bestand: één corrupt/onleesbaar bestand stopt de
  batch niet — de fout wordt gelogd, ingest gaat door met de rest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mutagen

from dj_engine import db as db_module
from dj_engine.analysis.essentia_extractor import analyze_audio
from dj_engine.analysis.mood_genre_models import classify_mood_genre
from dj_engine.enrichment.musicbrainz_client import MusicBrainzClient

logger = logging.getLogger(__name__)


def find_audio_files(folder_path: str, extensions: list[str]) -> list[Path]:
    """Loop recursief door folder_path, retourneer alle bestanden met een
    van de gegeven extensies (hoofdletterongevoelig), alfabetisch gesorteerd."""
    root = Path(folder_path)
    exts = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def read_tags(filepath: str) -> dict[str, str | None]:
    """Lees title/artist uit bestandstags met mutagen.

    Geeft None terug voor velden die ontbreken of bij een onleesbaar
    bestand — nooit fabriceren, en crasht nooit de caller.
    """
    try:
        audio_file = mutagen.File(filepath, easy=True)
    except mutagen.MutagenError as exc:
        logger.warning("Kon tags niet lezen uit %s: %s", filepath, exc)
        return {"title": None, "artist": None}

    if audio_file is None:
        logger.warning("Onherkend audioformaat (geen tag-ondersteuning) voor %s", filepath)
        return {"title": None, "artist": None}

    def _first(key: str) -> str | None:
        values = audio_file.get(key)
        return values[0] if values else None

    return {"title": _first("title"), "artist": _first("artist")}


def ingest_file(
    filepath: Path,
    config: dict[str, Any],
    mb_client: MusicBrainzClient | None,
) -> dict[str, Any] | None:
    """Analyseer en verrijk één bestand.

    Returns:
        Een track-dict klaar voor db.upsert_track(), of None als de
        audio-analyse mislukte (bestand wordt dan overgeslagen).
    """
    tags = read_tags(str(filepath))

    analysis = analyze_audio(str(filepath), sample_rate=config["audio"]["sample_rate"])
    if analysis["error"] is not None:
        logger.warning("Analyse overgeslagen voor %s: %s", filepath, analysis["error"])
        return None

    mood_genre = classify_mood_genre(str(filepath), config)

    mbid, mb_genre_tags = (None, None)
    if mb_client is not None:
        mbid, mb_genre_tags = mb_client.lookup_track(tags["title"], tags["artist"])

    # Genre: voorkeur voor het TensorFlow-genre-model (specifieker), val
    # terug op MusicBrainz-tags als het model niet beschikbaar is.
    genre = mood_genre["genre"] or mb_genre_tags

    return {
        "filepath": str(filepath),
        "title": tags["title"],
        "artist": tags["artist"],
        "mbid": mbid,
        "bpm": analysis["bpm"],
        "key": analysis["key"],
        "scale": analysis["scale"],
        "camelot": analysis["camelot"],
        "loudness": analysis["loudness"],
        "danceability": analysis["danceability"],
        "energy_raw": analysis["energy_raw"],
        "mood_happy": mood_genre["mood_happy"],
        "mood_sad": mood_genre["mood_sad"],
        "mood_aggressive": mood_genre["mood_aggressive"],
        "mood_relaxed": mood_genre["mood_relaxed"],
        "mood_party": mood_genre["mood_party"],
        "genre": genre,
    }


def ingest_folder(
    folder_path: str,
    conn,
    config: dict[str, Any],
    mb_client: MusicBrainzClient | None = None,
    progress: bool = True,
) -> dict[str, int]:
    """Scan `folder_path` recursief en analyseer/verrijk/sla nieuwe tracks op.

    Idempotent: bestanden waarvan het filepath al in de database staat
    worden overgeslagen (geen herberekening, geen duplicaten). Fouten per
    bestand stoppen de batch niet.

    Args:
        folder_path: map om recursief te scannen op audio-extensies.
        conn: open sqlite3-connectie (zie db.connect()).
        config: app-config (zie dj_engine.config.load_config()).
        mb_client: optionele MusicBrainzClient. None = geen enrichment
            (handig voor offline gebruik of tests).
        progress: toon een tqdm-voortgangsbalk.

    Returns:
        Statistieken: {"scanned", "skipped_existing", "ingested", "failed"}.
    """
    extensions = config["audio"]["extensions"]
    files = find_audio_files(folder_path, extensions)

    stats = {"scanned": len(files), "skipped_existing": 0, "ingested": 0, "failed": 0}

    iterator = files
    if progress:
        from tqdm import tqdm

        iterator = tqdm(files, desc="Ingesting", unit="track")

    for filepath in iterator:
        filepath_str = str(filepath)
        if db_module.track_exists(conn, filepath_str):
            stats["skipped_existing"] += 1
            continue

        try:
            track = ingest_file(filepath, config, mb_client)
        except Exception as exc:
            # Laatste vangnet: een onverwachte fout in één bestand mag de
            # rest van de batch niet laten stoppen.
            logger.error("Onverwachte fout bij verwerken van %s: %s", filepath, exc)
            track = None

        if track is None:
            stats["failed"] += 1
            continue

        db_module.upsert_track(conn, track)
        stats["ingested"] += 1

    db_module.normalize_energy(conn)
    logger.info(
        "Ingest klaar: %d gescand, %d nieuw, %d overgeslagen (bestond al), %d mislukt",
        stats["scanned"], stats["ingested"], stats["skipped_existing"], stats["failed"],
    )
    return stats
