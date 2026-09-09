"""Camelot Wheel: key/scale -> Camelot-code mapping + compatibiliteitsscore.

Gebruikt de standaard Camelot Wheel-indeling (zoals gebruikt door Mixed In
Key, Rekordbox, Serato e.d.): 1A-12A voor mineur, 1B-12B voor majeur, met
majeur/mineur-paren die op hetzelfde nummer relative major/minor zijn.
"""
from __future__ import annotations

import re

# Canonieke camelot-codes per toonsoort, met steun voor zowel kruizen (#)
# als mollen (b) omdat Essentia's KeyExtractor en tag-metadata beide kunnen
# leveren.
_MAJOR_TO_CAMELOT: dict[str, str] = {
    "C": "8B",
    "G": "9B",
    "D": "10B",
    "A": "11B",
    "E": "12B",
    "B": "1B",
    "F#": "2B", "Gb": "2B",
    "C#": "3B", "Db": "3B",
    "G#": "4B", "Ab": "4B",
    "D#": "5B", "Eb": "5B",
    "A#": "6B", "Bb": "6B",
    "F": "7B",
}

_MINOR_TO_CAMELOT: dict[str, str] = {
    "A": "8A",
    "E": "9A",
    "B": "10A",
    "F#": "11A", "Gb": "11A",
    "C#": "12A", "Db": "12A",
    "G#": "1A", "Ab": "1A",
    "D#": "2A", "Eb": "2A",
    "A#": "3A", "Bb": "3A",
    "F": "4A",
    "C": "5A",
    "G": "6A",
    "D": "7A",
}

_CAMELOT_CODE_RE = re.compile(r"^(?P<number>[1-9]|1[0-2])(?P<letter>[AB])$")

# Default compatibiliteitsscores, gespiegeld aan config.yaml's
# scoring.camelot_scores. Een expliciet meegegeven dict (bv. uit config)
# overrulet deze defaults.
DEFAULT_SCORES: dict[str, float] = {
    "same": 100,
    "adjacent_same_letter": 90,
    "relative_major_minor": 85,
    "two_step_same_letter": 75,
    "other": 30,
}


def key_to_camelot(key: str, scale: str) -> str | None:
    """Zet een (key, scale)-paar om naar een Camelot-code, bv. ("C", "major") -> "8B".

    Args:
        key: toonsoort-naam, bv. "C", "C#", "Db", "F#".
        scale: "major" of "minor" (hoofdletterongevoelig).

    Returns:
        Camelot-code als string, of None als key/scale niet herkend wordt
        (nooit gefabriceerd/geraden — de caller moet dit als "onbekend"
        behandelen).
    """
    if key is None or scale is None:
        return None

    normalized_key = key.strip().capitalize().replace("♭", "b").replace("♯", "#")
    normalized_scale = scale.strip().lower()

    if normalized_scale == "major":
        return _MAJOR_TO_CAMELOT.get(normalized_key)
    if normalized_scale == "minor":
        return _MINOR_TO_CAMELOT.get(normalized_key)
    return None


def _parse_camelot(code: str) -> tuple[int, str] | None:
    match = _CAMELOT_CODE_RE.match(code.strip().upper())
    if not match:
        return None
    return int(match.group("number")), match.group("letter")


def camelot_distance(
    a: str, b: str, scores: dict[str, float] | None = None
) -> float:
    """Compatibiliteitsscore (0-100) tussen twee Camelot-codes.

    Regels (met wrap-around van 12 naar 1):
    - Zelfde code: 100
    - ±1 stap, zelfde letter (bv. 8A -> 9A of 7A): 90
    - Zelfde nummer, andere letter (relative major/minor, bv. 8A -> 8B): 85
    - ±2 stappen, zelfde letter (energy boost/drop): 75
    - Overig: 30 (instelbaar via `scores["other"]`)

    Args:
        a, b: Camelot-codes, bv. "8A", "12B".
        scores: optionele override van de scoretabel (zie DEFAULT_SCORES).

    Raises:
        ValueError: als a of b geen geldige Camelot-code is.
    """
    table = {**DEFAULT_SCORES, **(scores or {})}

    parsed_a = _parse_camelot(a)
    parsed_b = _parse_camelot(b)
    if parsed_a is None:
        raise ValueError(f"Ongeldige Camelot-code: {a!r}")
    if parsed_b is None:
        raise ValueError(f"Ongeldige Camelot-code: {b!r}")

    number_a, letter_a = parsed_a
    number_b, letter_b = parsed_b

    if number_a == number_b and letter_a == letter_b:
        return table["same"]

    if letter_a == letter_b:
        # Kortste afstand op de wielring van 12 (wrap-around 12 <-> 1).
        diff = abs(number_a - number_b)
        step = min(diff, 12 - diff)
        if step == 1:
            return table["adjacent_same_letter"]
        if step == 2:
            return table["two_step_same_letter"]
        return table["other"]

    # Andere letter (A vs B)
    if number_a == number_b:
        return table["relative_major_minor"]

    return table["other"]
