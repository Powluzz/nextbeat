"""Tests voor dj_engine.recommend.llm_suggest.

parse_llm_response() en build_prompt() zijn puur en providerloos: geen
mocks nodig. suggest_next_llm() krijgt een gemockte provider (via
monkeypatch van get_provider) — geen echte netwerkcalls.
"""
from __future__ import annotations

import pytest

from dj_engine import db
from dj_engine.config import load_config
from dj_engine.enrichment.llm_providers import LLMProviderError, LLMResponse
from dj_engine.recommend.llm_suggest import build_prompt, parse_llm_response, suggest_next_llm


# --- build_prompt --------------------------------------------------------

def _track(**overrides):
    base = {"id": 1, "artist": "Artist A", "title": "Title A", "bpm": 128.0, "camelot": "8A", "genre": "Techno"}
    base.update(overrides)
    return base


def test_build_prompt_contains_current_and_candidates():
    current = _track(id=1, artist="Current Artist", title="Current Title")
    candidates = [_track(id=2, artist="Cand One"), _track(id=3, artist="Cand Two")]

    prompt = build_prompt(current, candidates, "build")

    assert "Current Artist" in prompt
    assert "Cand One" in prompt
    assert "Cand Two" in prompt
    assert "[id=2]" in prompt
    assert "[id=3]" in prompt
    assert "RICHTING: build" in prompt


def test_build_prompt_includes_json_format_instructions():
    prompt = build_prompt(_track(), [_track(id=2)], "hold")
    assert "JSON" in prompt
    assert "track_id" in prompt
    assert "gegrond_op_bron" in prompt


def test_build_prompt_invalid_direction_raises():
    with pytest.raises(ValueError):
        build_prompt(_track(), [_track(id=2)], "sideways")


def test_build_prompt_missing_fields_use_placeholder():
    current = {"id": 1, "artist": None, "title": None, "bpm": None, "camelot": None, "genre": None}
    prompt = build_prompt(current, [_track(id=2)], "ease")
    assert "?" in prompt  # geen crash op ontbrekende velden


# --- parse_llm_response ---------------------------------------------------

def test_parse_valid_json_array():
    text = '[{"track_id": 1, "reden": "Past goed", "gegrond_op_bron": true}]'
    result = parse_llm_response(text, valid_ids={1, 2})
    assert result == [{"track_id": 1, "reden": "Past goed", "gegrond_op_bron": True}]


def test_parse_strips_markdown_code_fence():
    text = '```json\n[{"track_id": 1, "reden": "ok", "gegrond_op_bron": false}]\n```'
    result = parse_llm_response(text, valid_ids={1})
    assert len(result) == 1
    assert result[0]["track_id"] == 1


def test_parse_extracts_array_from_surrounding_prose():
    text = 'Hier is mijn antwoord:\n[{"track_id": 1, "reden": "ok", "gegrond_op_bron": true}]\nHoop dat dit helpt!'
    result = parse_llm_response(text, valid_ids={1})
    assert len(result) == 1


def test_parse_filters_hallucinated_track_id():
    text = '[{"track_id": 999, "reden": "verzonnen", "gegrond_op_bron": false}]'
    result = parse_llm_response(text, valid_ids={1, 2})
    assert result == []


def test_parse_filters_only_hallucinated_keeps_valid():
    text = '[{"track_id": 1, "reden": "ok", "gegrond_op_bron": true}, {"track_id": 999, "reden": "nep", "gegrond_op_bron": false}]'
    result = parse_llm_response(text, valid_ids={1, 2})
    assert len(result) == 1
    assert result[0]["track_id"] == 1


def test_parse_skips_non_dict_items():
    text = '[{"track_id": 1, "reden": "ok", "gegrond_op_bron": true}, "garbage", 42]'
    result = parse_llm_response(text, valid_ids={1})
    assert len(result) == 1


def test_parse_completely_unparseable_returns_empty():
    result = parse_llm_response("dit is helemaal geen JSON", valid_ids={1})
    assert result == []


def test_parse_missing_optional_fields_get_defaults():
    text = '[{"track_id": 1}]'
    result = parse_llm_response(text, valid_ids={1})
    assert result == [{"track_id": 1, "reden": "", "gegrond_op_bron": False}]


def test_parse_non_int_track_id_skipped():
    text = '[{"track_id": "1", "reden": "string i.p.v. int", "gegrond_op_bron": false}]'
    result = parse_llm_response(text, valid_ids={1})
    assert result == []


def test_parse_empty_array_returns_empty():
    assert parse_llm_response("[]", valid_ids={1, 2}) == []


# --- suggest_next_llm ------------------------------------------------------

@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def config():
    return load_config()


def _insert(conn, **overrides):
    track = {
        "filepath": overrides.pop("filepath", f"/music/{overrides.get('title', 'x')}.mp3"),
        "title": "Untitled", "artist": "Unknown", "bpm": 128.0, "camelot": "8A",
    }
    track.update(overrides)
    return db.insert_track(conn, track)


def _patch_provider(monkeypatch, response: LLMResponse):
    class _FakeProvider:
        def complete(self, prompt, cfg):
            return response

    monkeypatch.setattr("dj_engine.recommend.llm_suggest.get_provider", lambda config: _FakeProvider())


def test_suggest_next_llm_happy_path(conn, config, monkeypatch):
    current_id = _insert(conn, title="current")
    cand_a = _insert(conn, title="cand_a")
    cand_b = _insert(conn, title="cand_b")

    _patch_provider(monkeypatch, LLMResponse(
        text=f'[{{"track_id": {cand_b}, "reden": "Goede match", "gegrond_op_bron": true}}, '
             f'{{"track_id": {cand_a}, "reden": "Ook prima", "gegrond_op_bron": false}}]',
        used_search=True,
        model="claude-sonnet-5",
    ))

    outcome = suggest_next_llm(conn, current_id, "build", config, top_n=10)

    assert outcome["used_search"] is True
    assert outcome["model"] == "claude-sonnet-5"
    assert [r["track"]["id"] for r in outcome["results"]] == [cand_b, cand_a]
    assert outcome["results"][0]["reden"] == "Goede match"
    assert outcome["results"][0]["gegrond_op_bron"] is True


def test_suggest_next_llm_respects_top_n(conn, config, monkeypatch):
    current_id = _insert(conn, title="current")
    ids = [_insert(conn, title=f"c{i}") for i in range(5)]

    items = ", ".join(f'{{"track_id": {i}, "reden": "x", "gegrond_op_bron": false}}' for i in ids)
    _patch_provider(monkeypatch, LLMResponse(text=f"[{items}]", used_search=False, model="m"))

    outcome = suggest_next_llm(conn, current_id, "hold", config, top_n=2)
    assert len(outcome["results"]) == 2


def test_suggest_next_llm_unknown_track_id_raises(conn, config, monkeypatch):
    _patch_provider(monkeypatch, LLMResponse(text="[]", used_search=False, model="m"))
    with pytest.raises(ValueError):
        suggest_next_llm(conn, 999, "build", config)


def test_suggest_next_llm_invalid_direction_raises(conn, config, monkeypatch):
    current_id = _insert(conn, title="current")
    _patch_provider(monkeypatch, LLMResponse(text="[]", used_search=False, model="m"))
    with pytest.raises(ValueError):
        suggest_next_llm(conn, current_id, "sideways", config)


def test_suggest_next_llm_no_candidates_returns_empty(conn, config, monkeypatch):
    current_id = _insert(conn, title="current")
    _patch_provider(monkeypatch, LLMResponse(text="[]", used_search=False, model="m"))

    outcome = suggest_next_llm(conn, current_id, "build", config)
    assert outcome["results"] == []


def test_suggest_next_llm_filters_hallucinated_candidate(conn, config, monkeypatch):
    current_id = _insert(conn, title="current")
    real_cand = _insert(conn, title="real")

    _patch_provider(monkeypatch, LLMResponse(
        text=f'[{{"track_id": 999999, "reden": "verzonnen", "gegrond_op_bron": false}}, '
             f'{{"track_id": {real_cand}, "reden": "echt", "gegrond_op_bron": true}}]',
        used_search=False, model="m",
    ))

    outcome = suggest_next_llm(conn, current_id, "build", config)
    assert len(outcome["results"]) == 1
    assert outcome["results"][0]["track"]["id"] == real_cand


def test_suggest_next_llm_provider_error_propagates(conn, config, monkeypatch):
    current_id = _insert(conn, title="current")
    _insert(conn, title="cand")

    class _FailingProvider:
        def complete(self, prompt, cfg):
            raise LLMProviderError("geen API-key")

    monkeypatch.setattr("dj_engine.recommend.llm_suggest.get_provider", lambda config: _FailingProvider())

    with pytest.raises(LLMProviderError, match="geen API-key"):
        suggest_next_llm(conn, current_id, "build", config)
