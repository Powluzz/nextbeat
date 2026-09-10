"""Achtergrond-verfijning van een suggestie-shortlist: vult `energy` en
mood/genre aan voor kandidaten die dat nog missen, via een process-pool.

CPU-bound Essentia/TensorFlow-werk parallelliseert niet zinvol over threads
(GIL + native-library-threading) — vandaar `ProcessPoolExecutor`. Gebruikt
expliciet de 'spawn'-start-methode (niet Linux' default 'fork'): forken
ná het laden van TensorFlow in het hoofdproces is een bekende bron van
hangs/crashes bij native libraries. Met 'spawn' start elke worker een
schone Python-interpreter, dus dat risico bestaat niet — essentia/TF wordt
sowieso lazy (pas in de worker) geïmporteerd.

bpm/key/camelot worden hier NOOIT herberekend of overschreven (zie
ingest/pipeline.py::analyze_track voor diezelfde regel bij losse
on-demand-analyse) — dit script vult alleen wat alleen onze eigen
Essentia/TF-analyse kan leveren: energy_raw en mood/genre.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from dj_engine import db as db_module
from dj_engine.enrichment.rekordbox_client import STREAMING_SCHEMES

logger = logging.getLogger(__name__)


def _limit_worker_threads() -> None:
    """Beperkt TensorFlow/Essentia's interne threading per workerproces.

    Gemeten (zie sessie-log): TF-inference gebruikt zonder deze limiet uit
    zichzelf ~2 cores/proces. Zonder deze cap zouden N parallelle workers
    ruim boven het aantal beschikbare cores uitkomen en elkaar vertragen.
    Moet vóór het (lazy) importeren van essentia gezet worden.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")


def _analyze_one(filepath: str, sample_rate: int, config: dict[str, Any]) -> dict[str, Any]:
    """Draait in een apart (spawned) workerproces."""
    _limit_worker_threads()
    # Een spawned proces heeft geen eigen logging-config (configure_logging()
    # draait alleen in het CLI-hoofdproces) — zonder dit print elke
    # analyze_audio()/classify_mood_genre()-waarschuwing hier ongeformatteerd
    # mee, bovenop de nette samenvatting die het hoofdproces al logt zodra
    # future.result() terugkomt. ERROR-level onderdrukt die dubbele ruis.
    logging.basicConfig(level=logging.ERROR)
    from dj_engine.analysis.essentia_extractor import analyze_audio
    from dj_engine.analysis.mood_genre_models import classify_mood_genre

    analysis = analyze_audio(filepath, sample_rate=sample_rate)
    if analysis["error"] is not None:
        mood_genre = {
            "mood_happy": None, "mood_sad": None, "mood_aggressive": None,
            "mood_relaxed": None, "mood_party": None, "genre": None,
        }
    else:
        mood_genre = classify_mood_genre(filepath, config)
    return {"analysis": analysis, "mood_genre": mood_genre}


def refine_candidates(
    conn,
    config: dict[str, Any],
    track_ids: list[int],
    n_workers: int | None = None,
) -> int:
    """Vul energy/mood/genre aan voor de tracks in `track_ids` die dat nog
    missen, parallel over een process-pool.

    Slaat tracks over die al volledig geanalyseerd zijn, en tracks zonder
    lokaal bestand (streaming-schemes zoals tidal://) — daar is niets te
    analyseren.

    Args:
        conn: open sqlite3-connectie.
        config: app-config (analysis_pool.workers, audio.sample_rate, ...).
        track_ids: kandidaat-track-id's (bv. de shortlist uit suggest_next()).
        n_workers: override voor het aantal workerprocessen.

    Returns:
        Aantal daadwerkelijk verfijnde (bijgewerkte) tracks.
    """
    candidates = []
    for track_id in track_ids:
        track = db_module.get_track(conn, track_id)
        if track is None:
            continue
        if track["filepath"].startswith(STREAMING_SCHEMES):
            continue
        if track.get("energy_raw") is not None and track.get("mood_happy") is not None:
            continue
        candidates.append(track)

    if not candidates:
        return 0

    workers = n_workers or config.get("analysis_pool", {}).get("workers", 4)
    sample_rate = config["audio"]["sample_rate"]
    ctx = multiprocessing.get_context("spawn")

    refined = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(_analyze_one, c["filepath"], sample_rate, config): c
            for c in candidates
        }
        for future in as_completed(futures):
            track = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error(
                    "Verfijning mislukt voor track %s (%s): %s",
                    track["id"], track["filepath"], exc,
                )
                continue

            analysis = result["analysis"]
            if analysis["error"] is not None:
                logger.warning(
                    "Verfijning overgeslagen voor track %s: %s", track["id"], analysis["error"]
                )
                continue

            mood_genre = result["mood_genre"]
            update: dict[str, Any] = {
                "filepath": track["filepath"],
                "loudness": analysis["loudness"],
                "danceability": analysis["danceability"],
                "energy_raw": analysis["energy_raw"],
                "mood_happy": mood_genre["mood_happy"],
                "mood_sad": mood_genre["mood_sad"],
                "mood_aggressive": mood_genre["mood_aggressive"],
                "mood_relaxed": mood_genre["mood_relaxed"],
                "mood_party": mood_genre["mood_party"],
            }
            if track.get("bpm") is None:
                update["bpm"] = analysis["bpm"]
            if track.get("camelot") is None:
                update["key"] = analysis["key"]
                update["scale"] = analysis["scale"]
                update["camelot"] = analysis["camelot"]
            if track.get("genre") is None:
                update["genre"] = mood_genre["genre"]

            db_module.upsert_track(conn, update)
            refined += 1

    if refined:
        # Direct herschalen (niet wachten op de periodieke drempel) zodat de
        # net verfijnde tracks meteen een echte energy-score hebben voor de
        # herscoring die hierna volgt.
        db_module.normalize_energy(conn)

    return refined
