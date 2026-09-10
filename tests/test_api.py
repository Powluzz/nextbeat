"""Tests voor dj_engine.api.main (FastAPI) via TestClient — geen echte
server nodig. Essentia-analysetests draaien op de echte (zelfgegenereerde)
audiofixtures; LLM-provider-calls zijn gemockt (geen netwerk).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dj_engine import db
from dj_engine.api.main import create_app
from dj_engine.config import load_config
from dj_engine.enrichment.llm_providers import LLMProviderError, LLMResponse

pytest.importorskip("essentia", reason="essentia is een optionele dependency")

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg["database"]["path"] = str(tmp_path / "test.db")
    cfg["models"]["directory"] = str(tmp_path / "no_models_here")
    cfg["analysis_pool"] = {"workers": 2}
    return cfg


@pytest.fixture
def conn(config):
    connection = db.connect(config["database"]["path"])
    yield connection
    connection.close()


@pytest.fixture
def client(config):
    app = create_app(config)
    return TestClient(app)


def _insert(conn, **overrides):
    track = {
        "filepath": overrides.pop("filepath", f"/music/{overrides.get('title', 'x')}.mp3"),
        "title": "Untitled", "artist": "Unknown", "bpm": 128.0, "camelot": "8A",
    }
    track.update(overrides)
    return db.insert_track(conn, track)


# --- index ---------------------------------------------------------------

def test_index_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "dj-engine" in response.text


# --- search ----------------------------------------------------------------

def test_search_by_q(client, conn):
    _insert(conn, filepath="/a.mp3", artist="Daft Punk", title="One More Time")
    _insert(conn, filepath="/b.mp3", artist="Justice", title="Genesis")

    response = client.get("/tracks/search", params={"q": "daft"})

    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["artist"] == "Daft Punk"


def test_search_empty_query_returns_all_up_to_limit(client, conn):
    for i in range(3):
        _insert(conn, filepath=f"/{i}.mp3")
    response = client.get("/tracks/search")
    assert len(response.json()) == 3


# --- get track --------------------------------------------------------

def test_get_track_found(client, conn):
    track_id = _insert(conn, title="Hello")
    response = client.get(f"/tracks/{track_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "Hello"


def test_get_track_not_found(client):
    response = client.get("/tracks/999")
    assert response.status_code == 404


# --- analyze -----------------------------------------------------------

def test_analyze_preserves_bpm_fills_energy_and_normalizes(client, conn, tmp_path):
    f = tmp_path / "track.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", f)
    track_id = _insert(conn, filepath=str(f), bpm=126.5, camelot="9A")

    response = client.post(f"/tracks/{track_id}/analyze")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["bpm"] == 126.5  # onaangeroerd
    assert body["camelot"] == "9A"
    assert body["energy_raw"] is not None
    assert body["energy"] is not None  # direct genormaliseerd (anders dan CLI's periodieke drempel)


def test_analyze_not_found(client):
    response = client.post("/tracks/999/analyze")
    assert response.status_code == 404


# --- suggest (engine) ----------------------------------------------------

def test_suggest_returns_ranked_results(client, conn):
    current_id = _insert(conn, title="current", bpm=128, camelot="8A")
    _insert(conn, filepath="/b.mp3", title="cand", bpm=128, camelot="8A")

    response = client.get(f"/tracks/{current_id}/suggest", params={"direction": "hold"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["direction"] == "hold"
    assert len(body["results"]) == 1
    assert set(body["results"][0]["breakdown"].keys()) == {"key", "bpm", "energy", "mood"}


def test_suggest_invalid_direction_returns_400(client, conn):
    current_id = _insert(conn, title="current")
    response = client.get(f"/tracks/{current_id}/suggest", params={"direction": "sideways"})
    assert response.status_code == 400


def test_suggest_unknown_track_returns_404(client):
    response = client.get("/tracks/999/suggest", params={"direction": "hold"})
    assert response.status_code == 404


# --- refine --------------------------------------------------------------

def test_refine_returns_refined_count_and_updated_suggestions(client, conn, tmp_path):
    f1 = tmp_path / "current.wav"
    f2 = tmp_path / "cand.wav"
    shutil.copy(FIXTURES / "click_128bpm.wav", f1)
    shutil.copy(FIXTURES / "tone_c_major.wav", f2)
    # energy al gezet op de huidige track (zoals na 'Analyseer nu' in de UI) --
    # anders blijft energy_score() neutraal (50) ongeacht wat refine oplevert,
    # want die vergelijkt altijd t.o.v. de huidige track (zie scoring.py).
    current_id = _insert(conn, filepath=str(f1), bpm=128, camelot="8A", energy=0.5)
    _insert(conn, filepath=str(f2), bpm=120, camelot="8B")

    response = client.post(f"/tracks/{current_id}/refine", params={"direction": "build"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["refined_count"] == 1
    assert body["suggestions"]["results"][0]["breakdown"]["energy"] != 50  # niet meer neutraal


# --- suggest-llm (gemockt) -----------------------------------------------

def test_suggest_llm_success(client, conn, monkeypatch):
    current_id = _insert(conn, title="current")
    cand_id = _insert(conn, filepath="/b.mp3", title="cand")

    class _FakeProvider:
        def complete(self, prompt, cfg):
            return LLMResponse(
                text=f'[{{"track_id": {cand_id}, "reden": "Past goed", "gegrond_op_bron": true}}]',
                used_search=True, model="claude-sonnet-5",
            )

    monkeypatch.setattr("dj_engine.recommend.llm_suggest.get_provider", lambda config: _FakeProvider())

    response = client.post(f"/tracks/{current_id}/suggest-llm", params={"direction": "build"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["used_search"] is True
    assert body["results"][0]["track"]["id"] == cand_id
    assert body["results"][0]["gegrond_op_bron"] is True


def test_suggest_llm_provider_error_returns_502(client, conn, monkeypatch):
    current_id = _insert(conn, title="current")
    _insert(conn, filepath="/b.mp3", title="cand")

    class _FailingProvider:
        def complete(self, prompt, cfg):
            raise LLMProviderError("geen API-key")

    monkeypatch.setattr("dj_engine.recommend.llm_suggest.get_provider", lambda config: _FailingProvider())

    response = client.post(f"/tracks/{current_id}/suggest-llm", params={"direction": "build"})

    assert response.status_code == 502
    assert "geen API-key" in response.json()["detail"]


def test_suggest_llm_unknown_track_returns_404(client):
    response = client.post("/tracks/999/suggest-llm", params={"direction": "build"})
    assert response.status_code == 404


# --- transitions --------------------------------------------------------

def test_log_transition_success(client, conn):
    a = _insert(conn, filepath="/a.mp3")
    b = _insert(conn, filepath="/b.mp3")

    response = client.post("/transitions", json={"from_track_id": a, "to_track_id": b, "direction": "build"})

    assert response.status_code == 201
    assert "id" in response.json()

    transitions = db.get_transitions(conn)
    assert len(transitions) == 1
    assert transitions[0]["direction"] == "build"


def test_log_transition_unknown_from_track_404(client, conn):
    b = _insert(conn, filepath="/b.mp3")
    response = client.post("/transitions", json={"from_track_id": 999, "to_track_id": b, "direction": "hold"})
    assert response.status_code == 404


def test_log_transition_unknown_to_track_404(client, conn):
    a = _insert(conn, filepath="/a.mp3")
    response = client.post("/transitions", json={"from_track_id": a, "to_track_id": 999, "direction": "hold"})
    assert response.status_code == 404


def test_log_transition_with_rating(client, conn):
    a = _insert(conn, filepath="/a.mp3")
    b = _insert(conn, filepath="/b.mp3")
    response = client.post(
        "/transitions", json={"from_track_id": a, "to_track_id": b, "direction": "ease", "rating": 5}
    )
    assert response.status_code == 201
    transitions = db.get_transitions(conn)
    assert transitions[0]["rating"] == 5


# --- settings/llm + settings/api-key --------------------------------------

def test_get_llm_settings_key_not_configured(client, config, monkeypatch):
    monkeypatch.delenv(config["llm_suggest"]["api_key_env_var"], raising=False)

    response = client.get("/settings/llm")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "anthropic"
    assert body["api_key_env_var"] == config["llm_suggest"]["api_key_env_var"]
    assert body["api_key_configured"] is False


def test_get_llm_settings_key_configured(client, config, monkeypatch):
    monkeypatch.setenv(config["llm_suggest"]["api_key_env_var"], "sk-test")
    response = client.get("/settings/llm")
    assert response.json()["api_key_configured"] is True


def test_set_api_key_writes_env_file_and_updates_process_env(client, tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    var_name = config["llm_suggest"]["api_key_env_var"]
    monkeypatch.delenv(var_name, raising=False)

    response = client.post("/settings/api-key", json={"api_key": "sk-newkey"})

    assert response.status_code == 204
    assert (tmp_path / ".env").read_text().strip() == f"{var_name}=sk-newkey"
    assert os.environ[var_name] == "sk-newkey"


def test_set_api_key_reflected_in_settings_afterward(client, tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    var_name = config["llm_suggest"]["api_key_env_var"]
    monkeypatch.delenv(var_name, raising=False)

    assert client.get("/settings/llm").json()["api_key_configured"] is False
    client.post("/settings/api-key", json={"api_key": "sk-newkey"})
    assert client.get("/settings/llm").json()["api_key_configured"] is True


def test_set_api_key_custom_var_name(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MY_CUSTOM_KEY", raising=False)

    response = client.post("/settings/api-key", json={"api_key": "abc", "var_name": "MY_CUSTOM_KEY"})

    assert response.status_code == 204
    assert os.environ["MY_CUSTOM_KEY"] == "abc"


def test_set_api_key_empty_string_returns_400(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = client.post("/settings/api-key", json={"api_key": "   "})
    assert response.status_code == 400


def test_set_api_key_then_llm_call_works_without_restart(client, conn, tmp_path, monkeypatch, config):
    """Kern van deze feature: een net via de UI gezette key moet meteen
    bruikbaar zijn in dezelfde, al lopende serverinstantie — geen herstart."""
    monkeypatch.chdir(tmp_path)
    var_name = config["llm_suggest"]["api_key_env_var"]
    monkeypatch.delenv(var_name, raising=False)

    current_id = _insert(conn, title="current")
    cand_id = _insert(conn, filepath="/b.mp3", title="cand")

    # Vóór het zetten van de key: provider zou LLMProviderError geven
    # ("Geen API-key gevonden") — hier gemockt zodat we alleen het
    # environment-doorwerkingseffect testen, niet de provider zelf opnieuw.
    class _FakeProvider:
        def complete(self, prompt, cfg):
            assert os.environ.get(var_name) == "sk-live-test"  # bewijs: al gezien door de provider-call
            return LLMResponse(text=f'[{{"track_id": {cand_id}, "reden": "ok", "gegrond_op_bron": false}}]',
                                used_search=False, model="m")

    monkeypatch.setattr("dj_engine.recommend.llm_suggest.get_provider", lambda config: _FakeProvider())

    set_key_response = client.post("/settings/api-key", json={"api_key": "sk-live-test"})
    assert set_key_response.status_code == 204

    suggest_response = client.post(f"/tracks/{current_id}/suggest-llm", params={"direction": "build"})
    assert suggest_response.status_code == 200, suggest_response.text
