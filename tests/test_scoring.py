"""Tests voor dj_engine.recommend.scoring."""
from __future__ import annotations

import pytest

from dj_engine.recommend.scoring import (
    NEUTRAL_SCORE,
    bpm_score,
    energy_score,
    key_score,
    mood_score,
)


# --- key_score --------------------------------------------------------

def test_key_score_matches_camelot_distance():
    assert key_score("8A", "8A") == 100
    assert key_score("8A", "9A") == 90
    assert key_score("8A", "8B") == 85
    assert key_score("8A", "10A") == 75
    assert key_score("8A", "3A") == 30


def test_key_score_missing_data_returns_neutral():
    assert key_score(None, "8A") == NEUTRAL_SCORE
    assert key_score("8A", None) == NEUTRAL_SCORE
    assert key_score(None, None) == NEUTRAL_SCORE


def test_key_score_invalid_code_returns_neutral():
    assert key_score("garbage", "8A") == NEUTRAL_SCORE


def test_key_score_custom_scores_table():
    assert key_score("8A", "3A", scores={"other": 0}) == 0


# --- bpm_score --------------------------------------------------------

def test_bpm_score_identical_is_100():
    assert bpm_score(128.0, 128.0) == 100.0


def test_bpm_score_linear_falloff():
    # 8% van 128 = 10.24; 4% afwijking -> halverwege
    assert bpm_score(128.0, 128.0 * 1.04, max_deviation_pct=8.0) == pytest.approx(50.0, abs=0.5)


def test_bpm_score_at_max_deviation_is_zero():
    assert bpm_score(128.0, 128.0 * 1.08, max_deviation_pct=8.0) == pytest.approx(0.0, abs=0.1)


def test_bpm_score_beyond_max_deviation_is_zero():
    assert bpm_score(128.0, 160.0, max_deviation_pct=8.0) == 0.0


def test_bpm_score_symmetric_direction():
    up = bpm_score(128.0, 132.0, max_deviation_pct=8.0)
    down = bpm_score(128.0, 124.0, max_deviation_pct=8.0)
    assert up == pytest.approx(down, abs=1.0)


def test_bpm_score_missing_data_returns_neutral():
    assert bpm_score(None, 128.0) == NEUTRAL_SCORE
    assert bpm_score(128.0, None) == NEUTRAL_SCORE


def test_bpm_score_zero_bpm_a_returns_neutral():
    assert bpm_score(0.0, 128.0) == NEUTRAL_SCORE


def test_bpm_score_custom_max_deviation():
    # 10% afwijking, max_deviation_pct=20 -> nog steeds positieve score
    assert bpm_score(100.0, 110.0, max_deviation_pct=20.0) == pytest.approx(50.0, abs=0.5)


# --- energy_score --------------------------------------------------------

def test_energy_score_build_rewards_increase():
    high = energy_score(0.5, 0.75, "build")  # delta=0.25, binnen plateau (center 0.25)
    same = energy_score(0.5, 0.5, "build")   # delta=0, buiten plateau
    assert high == 100.0
    assert same < high


def test_energy_score_ease_rewards_decrease():
    lower = energy_score(0.5, 0.25, "ease")  # delta=-0.25
    same = energy_score(0.5, 0.5, "ease")
    assert lower == 100.0
    assert same < lower


def test_energy_score_hold_rewards_similar_energy():
    same = energy_score(0.5, 0.5, "hold")
    far = energy_score(0.5, 0.9, "hold")
    assert same == 100.0
    assert far < same


def test_energy_score_surprise_is_lenient():
    # Plateau dekt vrijwel het hele bereik -> altijd hoge score
    for delta_target in [0.0, 0.3, -0.3, 0.5]:
        score = energy_score(0.5, 0.5 + delta_target, "surprise")
        assert score >= 50.0


def test_energy_score_missing_data_returns_neutral():
    assert energy_score(None, 0.5, "build") == NEUTRAL_SCORE
    assert energy_score(0.5, None, "build") == NEUTRAL_SCORE


def test_energy_score_invalid_direction_raises():
    with pytest.raises(ValueError):
        energy_score(0.5, 0.5, "sideways")


def test_energy_score_custom_targets():
    custom = {"build": {"center": 0.5, "plateau": 0.0, "falloff": 0.5}}
    assert energy_score(0.0, 0.5, "build", targets=custom) == 100.0


# --- mood_score --------------------------------------------------------

def _track(**overrides):
    base = {
        "mood_happy": 0.5, "mood_sad": 0.5, "mood_aggressive": 0.5,
        "mood_relaxed": 0.5, "mood_party": 0.5, "genre": None,
    }
    base.update(overrides)
    return base


def test_mood_score_identical_mood_vectors_is_100():
    a = _track()
    b = _track()
    assert mood_score(a, b) == pytest.approx(100.0)


def test_mood_score_opposite_mood_vectors_is_low():
    a = _track(mood_happy=0.0, mood_sad=0.0, mood_aggressive=0.0, mood_relaxed=0.0, mood_party=0.0)
    b = _track(mood_happy=1.0, mood_sad=1.0, mood_aggressive=1.0, mood_relaxed=1.0, mood_party=1.0)
    assert mood_score(a, b) == pytest.approx(0.0, abs=0.01)


def test_mood_score_missing_mood_data_falls_back_to_genre():
    a = _track(mood_happy=None, genre="Techno")
    b = _track(genre="Techno,House")
    assert mood_score(a, b) == 100.0  # genre overlap, geen mood-vector beschikbaar


def test_mood_score_no_data_at_all_returns_neutral():
    a = _track(mood_happy=None, mood_sad=None, mood_aggressive=None, mood_relaxed=None, mood_party=None)
    b = _track(mood_happy=None, mood_sad=None, mood_aggressive=None, mood_relaxed=None, mood_party=None)
    assert mood_score(a, b) == NEUTRAL_SCORE


def test_mood_score_combines_mood_and_genre():
    a = _track(genre="Techno")
    b = _track(genre="Techno")
    combined = mood_score(a, b)
    c = _track(genre="House")
    mismatched_genre = mood_score(a, c)
    assert combined > mismatched_genre  # gedeeld genre moet score verhogen


def test_mood_score_genre_case_insensitive_and_multi_valued():
    a = _track(genre="Deep House, Techno")
    b = _track(genre="techno")
    assert mood_score(a, b) == 100.0
