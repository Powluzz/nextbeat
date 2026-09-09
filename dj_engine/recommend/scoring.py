"""Losse scorefuncties (0-100) per dimensie: key, bpm, energy, mood.

Elke functie retourneert een neutrale 50.0 als de benodigde data ontbreekt
(bv. camelot/bpm/mood nog niet geanalyseerd of modellen niet geladen) —
zo domineert een lege dimensie de ranking niet in het voor- of nadeel van
een track, en wordt er nooit data gefabriceerd om een score te forceren.
"""
from __future__ import annotations

import math
from typing import Any

from dj_engine.analysis.camelot import camelot_distance

NEUTRAL_SCORE = 50.0

MOOD_DIMENSIONS = [
    "mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party",
]

# Relatief gewicht van mood-vector-similarity t.o.v. genre-overlap binnen
# mood_score(), gebruikt wanneer beide beschikbaar zijn.
_MOOD_VECTOR_WEIGHT = 0.7
_GENRE_WEIGHT = 0.3
_GENRE_MATCH_SCORE = 100.0
_GENRE_MISMATCH_SCORE = 30.0

# Default doel-energieverschuiving per richting; normaal overschreven door
# config.yaml -> scoring.energy_targets (zie dj_engine.config).
DEFAULT_ENERGY_TARGETS: dict[str, dict[str, float]] = {
    "build": {"center": 0.25, "plateau": 0.10, "falloff": 0.50},
    "hold": {"center": 0.0, "plateau": 0.05, "falloff": 0.35},
    "ease": {"center": -0.25, "plateau": 0.10, "falloff": 0.50},
    "surprise": {"center": 0.0, "plateau": 1.0, "falloff": 0.50},
}

VALID_DIRECTIONS = ("build", "hold", "ease", "surprise")


def key_score(
    camelot_a: str | None,
    camelot_b: str | None,
    scores: dict[str, float] | None = None,
) -> float:
    """Camelot-compatibiliteit tussen twee tracks (0-100).

    Dunne wrapper rond analysis.camelot.camelot_distance(). Retourneert
    NEUTRAL_SCORE als een van beide camelot-codes ontbreekt of ongeldig is.
    """
    if not camelot_a or not camelot_b:
        return NEUTRAL_SCORE
    try:
        return float(camelot_distance(camelot_a, camelot_b, scores=scores))
    except ValueError:
        return NEUTRAL_SCORE


def bpm_score(
    bpm_a: float | None,
    bpm_b: float | None,
    max_deviation_pct: float = 8.0,
) -> float:
    """BPM-compatibiliteit (0-100), lineair aflopend tot 0 bij
    `max_deviation_pct` procentuele afwijking t.o.v. bpm_a (de huidige track).

    Retourneert NEUTRAL_SCORE als een van beide BPM's ontbreekt.
    """
    if bpm_a is None or bpm_b is None or bpm_a <= 0:
        return NEUTRAL_SCORE

    pct_diff = abs(bpm_a - bpm_b) / bpm_a * 100.0
    if pct_diff >= max_deviation_pct:
        return 0.0
    return 100.0 * (1.0 - pct_diff / max_deviation_pct)


def _trapezoid_score(value: float, center: float, plateau: float, falloff: float) -> float:
    """100 binnen `plateau` van `center`, lineair naar 0 over `falloff` daarna."""
    dist = abs(value - center)
    if dist <= plateau:
        return 100.0
    if falloff <= 0 or dist >= plateau + falloff:
        return 0.0
    return 100.0 * (1.0 - (dist - plateau) / falloff)


def energy_score(
    energy_a: float | None,
    energy_b: float | None,
    direction: str,
    targets: dict[str, dict[str, float]] | None = None,
) -> float:
    """Beloont een energieverschuiving (energy_b - energy_a) die past bij
    de gekozen richting (0-100). Zie config.yaml -> scoring.energy_targets
    voor de betekenis van center/plateau/falloff per richting.

    Retourneert NEUTRAL_SCORE als een van beide energiewaarden ontbreekt.
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Onbekende richting: {direction!r} (verwacht: {VALID_DIRECTIONS})")
    if energy_a is None or energy_b is None:
        return NEUTRAL_SCORE

    target = (targets or DEFAULT_ENERGY_TARGETS)[direction]
    delta = energy_b - energy_a
    return _trapezoid_score(delta, target["center"], target["plateau"], target["falloff"])


def mood_score(track_a: dict[str, Any], track_b: dict[str, Any]) -> float:
    """Similariteit in mood-profiel + genre-overlap tussen twee tracks (0-100).

    - Mood-similarity: 1 - genormaliseerde euclidische afstand over de 5
      mood_*-dimensies (alleen berekend als beide tracks alle 5 waarden hebben).
    - Genre-overlap: 100 bij minstens één gedeeld genre-label, anders 30
      (comma-gescheiden strings, hoofdletterongevoelig vergeleken).
    - Als beide beschikbaar zijn: 70% mood-similarity + 30% genre-overlap.
    - Als er niets beschikbaar is (geen modellen, geen genre-tags):
      NEUTRAL_SCORE — nooit fabriceren.
    """
    mood_similarity = _mood_vector_similarity(track_a, track_b)
    genre_match = _genre_overlap(track_a.get("genre"), track_b.get("genre"))

    if mood_similarity is not None and genre_match is not None:
        return _MOOD_VECTOR_WEIGHT * mood_similarity + _GENRE_WEIGHT * genre_match
    if mood_similarity is not None:
        return mood_similarity
    if genre_match is not None:
        return genre_match
    return NEUTRAL_SCORE


def _mood_vector_similarity(track_a: dict[str, Any], track_b: dict[str, Any]) -> float | None:
    values_a = [track_a.get(dim) for dim in MOOD_DIMENSIONS]
    values_b = [track_b.get(dim) for dim in MOOD_DIMENSIONS]
    if any(v is None for v in values_a) or any(v is None for v in values_b):
        return None

    squared_diffs = sum((a - b) ** 2 for a, b in zip(values_a, values_b))
    distance = math.sqrt(squared_diffs)
    max_distance = math.sqrt(len(MOOD_DIMENSIONS))  # alle dims maximaal 1 uit elkaar
    return 100.0 * (1.0 - distance / max_distance)


def _genre_overlap(genre_a: str | None, genre_b: str | None) -> float | None:
    if not genre_a or not genre_b:
        return None
    set_a = {g.strip().lower() for g in genre_a.split(",") if g.strip()}
    set_b = {g.strip().lower() for g in genre_b.split(",") if g.strip()}
    if not set_a or not set_b:
        return None
    return _GENRE_MATCH_SCORE if set_a & set_b else _GENRE_MISMATCH_SCORE
