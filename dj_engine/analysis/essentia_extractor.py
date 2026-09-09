"""Essentia-gebaseerde audio-analyse: BPM, key, loudness, danceability,
ruwe energie-feature.

Faalt nooit hard op een individueel bestand: corrupte/onleesbare bestanden
worden gelogd en als dict met een gevuld `error`-veld teruggegeven, zodat
een ingest-batch (zie ingest/pipeline.py) kan doorgaan met de rest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dj_engine.analysis.camelot import key_to_camelot

logger = logging.getLogger(__name__)

# Frequentieband gebruikt voor de 'energy_raw'-feature: verhouding
# hoogfrequente spectrale energie (hi-hats, ruis, distortie — doorgaans
# sterker aanwezig in energieke/percussieve tracks) t.o.v. totale energie
# per frame. Zie config.yaml['audio']['sample_rate'] voor de sample rate.
ENERGY_BAND_START_HZ = 1500.0
ENERGY_BAND_STOP_HZ = 11000.0
FRAME_SIZE = 2048
HOP_SIZE = 1024

# Minimale audiolengte (seconden) om een analysepoging zinvol te achten.
MIN_DURATION_S = 1.0

# Onder deze detectiezekerheid wordt een waarschuwing gelogd (de waarde
# wordt wél teruggegeven — nooit fabriceren, wel transparant over kwaliteit).
LOW_BPM_CONFIDENCE_THRESHOLD = 1.0
LOW_KEY_STRENGTH_THRESHOLD = 0.3


def _empty_result(filepath: str, error: str) -> dict[str, Any]:
    return {
        "filepath": filepath,
        "bpm": None,
        "bpm_confidence": None,
        "key": None,
        "scale": None,
        "key_strength": None,
        "camelot": None,
        "loudness": None,
        "danceability": None,
        "energy_raw": None,
        "error": error,
    }


def analyze_audio(filepath: str, sample_rate: int = 44100) -> dict[str, Any]:
    """Analyseer één audiobestand met Essentia.

    Retourneert altijd een dict (nooit een exception naar de caller):
    - Bij succes: alle analysevelden gevuld, `error` is None.
    - Bij falen (corrupt bestand, te kort, onbekend formaat, ontbrekend
      bestand): alle analysevelden None, `error` bevat een leesbare
      boodschap. De caller logt dit en slaat het bestand over.

    Velden in de returned dict:
        bpm, bpm_confidence: RhythmExtractor2013(method="multifeature")
        key, scale, key_strength: KeyExtractor()
        camelot: afgeleid via analysis.camelot.key_to_camelot()
        loudness: Loudness()
        danceability: Danceability()
        energy_raw: ongenormaliseerde energie-feature (RMS + hoogfrequent-
            energieratio); wordt pas over de hele bibliotheek 0-1
            genormaliseerd door de ingest-pipeline.
    """
    try:
        import essentia.standard as es
    except ImportError as exc:
        msg = f"essentia is niet geïnstalleerd: {exc}"
        logger.error(msg)
        return _empty_result(filepath, msg)

    if not Path(filepath).exists():
        msg = f"Bestand niet gevonden: {filepath}"
        logger.warning(msg)
        return _empty_result(filepath, msg)

    try:
        audio = es.MonoLoader(filename=str(filepath), sampleRate=sample_rate)()
    except Exception as exc:
        # Essentia gooit RuntimeError voor corrupte/onbekende/lege bestanden.
        msg = f"Kon audio niet laden ({filepath}): {exc}"
        logger.warning(msg)
        return _empty_result(filepath, msg)

    duration_s = len(audio) / float(sample_rate) if sample_rate else 0.0
    if len(audio) == 0 or duration_s < MIN_DURATION_S:
        msg = f"Audio te kort of leeg ({duration_s:.2f}s): {filepath}"
        logger.warning(msg)
        return _empty_result(filepath, msg)

    try:
        bpm, _ticks, bpm_confidence, _estimates, _bpm_intervals = es.RhythmExtractor2013(
            method="multifeature"
        )(audio)

        key, scale, key_strength = es.KeyExtractor()(audio)
        loudness = es.Loudness()(audio)
        danceability, _dfa = es.Danceability()(audio)
        energy_raw = _compute_raw_energy(es, audio, sample_rate)

        camelot = key_to_camelot(key, scale)
        if camelot is None:
            logger.warning(
                "Onbekende toonsoort '%s %s' in %s — camelot blijft NULL",
                key, scale, filepath,
            )
        if bpm_confidence is not None and bpm_confidence < LOW_BPM_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Lage BPM-detectiezekerheid (%.2f) voor %s — bpm=%.1f mogelijk onbetrouwbaar",
                bpm_confidence, filepath, bpm,
            )
        if key_strength is not None and key_strength < LOW_KEY_STRENGTH_THRESHOLD:
            logger.warning(
                "Lage key-detectiezekerheid (%.2f) voor %s — key=%s %s mogelijk onbetrouwbaar",
                key_strength, filepath, key, scale,
            )

        return {
            "filepath": filepath,
            "bpm": float(bpm),
            "bpm_confidence": float(bpm_confidence),
            "key": key,
            "scale": scale,
            "key_strength": float(key_strength),
            "camelot": camelot,
            "loudness": float(loudness),
            "danceability": float(danceability),
            "energy_raw": energy_raw,
            "error": None,
        }
    except Exception as exc:
        msg = f"Analysefout bij {filepath}: {exc}"
        logger.error(msg)
        return _empty_result(filepath, msg)


def _compute_raw_energy(es_module, audio: np.ndarray, sample_rate: int) -> float:
    """Ongenormaliseerde energie-feature: gemiddelde RMS + gemiddelde
    hoogfrequent-energieratio over frames.

    Wordt pas 0-1 genormaliseerd op bibliotheekniveau (min-max over alle
    tracks, zie ingest/pipeline.py::normalize_energy) omdat dat een
    zinvollere, relatieve schaal geeft binnen de eigen collectie dan een
    vaste absolute grens die nergens op gekalibreerd is.
    """
    windowing = es_module.Windowing(type="hann")
    spectrum = es_module.Spectrum()
    rms = es_module.RMS()
    band_ratio = es_module.EnergyBandRatio(
        sampleRate=sample_rate,
        startFrequency=ENERGY_BAND_START_HZ,
        stopFrequency=ENERGY_BAND_STOP_HZ,
    )

    rms_values = []
    band_values = []
    for frame in es_module.FrameGenerator(
        audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=True
    ):
        rms_values.append(rms(frame))
        band_values.append(band_ratio(spectrum(windowing(frame))))

    if not rms_values:
        return 0.0

    avg_rms = float(np.mean(rms_values))
    avg_band_ratio = float(np.mean(band_values))
    return avg_rms + avg_band_ratio
