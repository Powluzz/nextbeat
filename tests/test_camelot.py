"""Tests voor dj_engine.analysis.camelot."""
from __future__ import annotations

import pytest

from dj_engine.analysis.camelot import (
    DEFAULT_SCORES,
    camelot_distance,
    key_to_camelot,
)


# --- key_to_camelot -----------------------------------------------------

@pytest.mark.parametrize(
    "key,scale,expected",
    [
        ("C", "major", "8B"),
        ("G", "major", "9B"),
        ("D", "major", "10B"),
        ("A", "major", "11B"),
        ("E", "major", "12B"),
        ("B", "major", "1B"),
        ("F#", "major", "2B"),
        ("Gb", "major", "2B"),
        ("C#", "major", "3B"),
        ("Db", "major", "3B"),
        ("G#", "major", "4B"),
        ("Ab", "major", "4B"),
        ("D#", "major", "5B"),
        ("Eb", "major", "5B"),
        ("A#", "major", "6B"),
        ("Bb", "major", "6B"),
        ("F", "major", "7B"),
        ("A", "minor", "8A"),
        ("E", "minor", "9A"),
        ("B", "minor", "10A"),
        ("F#", "minor", "11A"),
        ("Gb", "minor", "11A"),
        ("C#", "minor", "12A"),
        ("Db", "minor", "12A"),
        ("G#", "minor", "1A"),
        ("Ab", "minor", "1A"),
        ("D#", "minor", "2A"),
        ("Eb", "minor", "2A"),
        ("A#", "minor", "3A"),
        ("Bb", "minor", "3A"),
        ("F", "minor", "4A"),
        ("C", "minor", "5A"),
        ("G", "minor", "6A"),
        ("D", "minor", "7A"),
    ],
)
def test_key_to_camelot_all_24_keys(key, scale, expected):
    assert key_to_camelot(key, scale) == expected


def test_key_to_camelot_case_insensitive_scale():
    assert key_to_camelot("C", "Major") == "8B"
    assert key_to_camelot("c", "MAJOR") == "8B"


def test_key_to_camelot_unknown_key_returns_none():
    assert key_to_camelot("H", "major") is None


def test_key_to_camelot_unknown_scale_returns_none():
    assert key_to_camelot("C", "dorian") is None


def test_key_to_camelot_none_input_returns_none():
    assert key_to_camelot(None, "major") is None
    assert key_to_camelot("C", None) is None


def test_all_camelot_codes_1_to_12_covered_both_letters():
    codes = set(_get_all_codes())
    expected = {f"{n}{letter}" for n in range(1, 13) for letter in ("A", "B")}
    assert codes == expected


def _get_all_codes():
    from dj_engine.analysis.camelot import _MAJOR_TO_CAMELOT, _MINOR_TO_CAMELOT
    return set(_MAJOR_TO_CAMELOT.values()) | set(_MINOR_TO_CAMELOT.values())


# --- camelot_distance -----------------------------------------------------

def test_same_code_scores_100():
    assert camelot_distance("8A", "8A") == 100
    assert camelot_distance("8B", "8B") == 100


def test_adjacent_same_letter_scores_90():
    assert camelot_distance("8A", "9A") == 90
    assert camelot_distance("8A", "7A") == 90
    assert camelot_distance("8B", "9B") == 90


def test_relative_major_minor_scores_85():
    assert camelot_distance("8A", "8B") == 85
    assert camelot_distance("8B", "8A") == 85


def test_two_step_same_letter_scores_75():
    assert camelot_distance("8A", "10A") == 75
    assert camelot_distance("8A", "6A") == 75


def test_other_scores_30():
    # 8A -> 3A: verschil van 5, geen enkele bekende regel van toepassing
    assert camelot_distance("8A", "3A") == 30
    # Andere letter, andere nummer: geen relative major/minor, geen adjacency
    assert camelot_distance("8A", "9B") == 30


def test_wrap_around_12_to_1_adjacent():
    assert camelot_distance("12A", "1A") == 90
    assert camelot_distance("1A", "12A") == 90
    assert camelot_distance("12B", "1B") == 90


def test_wrap_around_12_to_1_two_step():
    assert camelot_distance("12A", "2A") == 75
    assert camelot_distance("11A", "1A") == 75


def test_wrap_around_symmetry_1_and_12_extremes():
    # 1 -> 11 zou via wrap-around ook een 2-stap moeten zijn (1->12->11)
    assert camelot_distance("1A", "11A") == 75


def test_camelot_distance_is_symmetric():
    for a, b in [("8A", "9A"), ("8A", "8B"), ("8A", "10A"), ("8A", "3A")]:
        assert camelot_distance(a, b) == camelot_distance(b, a)


def test_camelot_distance_invalid_code_raises():
    with pytest.raises(ValueError):
        camelot_distance("13A", "8A")
    with pytest.raises(ValueError):
        camelot_distance("8A", "0A")
    with pytest.raises(ValueError):
        camelot_distance("8C", "8A")
    with pytest.raises(ValueError):
        camelot_distance("garbage", "8A")


def test_camelot_distance_case_insensitive():
    assert camelot_distance("8a", "9a") == 90


def test_camelot_distance_custom_scores_override():
    custom = {"other": 0}
    assert camelot_distance("8A", "3A", scores=custom) == 0
    # niet-overriden entries blijven default
    assert camelot_distance("8A", "8A", scores=custom) == DEFAULT_SCORES["same"]
