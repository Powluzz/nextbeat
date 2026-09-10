"""Tests voor dj_engine.enrichment.llm_providers.

Alle netwerk-/SDK-calls zijn gemockt — er gaat nooit een echt verzoek naar
Anthropic of een OpenAI-compatibele API uit tijdens tests.
"""
from __future__ import annotations

import json
import sys
import types
import urllib.error

import pytest

from dj_engine.enrichment.llm_providers import LLMProviderError, get_provider
from dj_engine.enrichment.llm_providers.openai_compatible_provider import OpenAICompatibleProvider


# --- get_provider (factory) --------------------------------------------------

def test_get_provider_anthropic():
    from dj_engine.enrichment.llm_providers.anthropic_provider import AnthropicProvider
    provider = get_provider({"llm_suggest": {"provider": "anthropic"}})
    assert isinstance(provider, AnthropicProvider)


def test_get_provider_openai_compatible():
    provider = get_provider({"llm_suggest": {"provider": "openai_compatible"}})
    assert isinstance(provider, OpenAICompatibleProvider)


def test_get_provider_default_is_anthropic():
    from dj_engine.enrichment.llm_providers.anthropic_provider import AnthropicProvider
    provider = get_provider({"llm_suggest": {}})
    assert isinstance(provider, AnthropicProvider)


def test_get_provider_unknown_raises():
    with pytest.raises(LLMProviderError, match="Onbekende provider"):
        get_provider({"llm_suggest": {"provider": "does-not-exist"}})


# --- AnthropicProvider ---------------------------------------------------

class _FakeAnthropicError(Exception):
    pass


class _FakeAuthenticationError(_FakeAnthropicError):
    pass


class _FakeRateLimitError(_FakeAnthropicError):
    pass


class _FakeAPIConnectionError(_FakeAnthropicError):
    pass


class _FakeAPIStatusError(_FakeAnthropicError):
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class _FakeBlock:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class _FakeResponse:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeMessagesAPI:
    def __init__(self, create_fn):
        self._create_fn = create_fn
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._create_fn(**kwargs)


class _FakeAnthropicClient:
    def __init__(self, create_fn, api_key=None):
        self.api_key = api_key
        self.messages = _FakeMessagesAPI(create_fn)


def _install_fake_anthropic(monkeypatch, create_fn):
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda api_key=None: _FakeAnthropicClient(create_fn, api_key=api_key)
    mod.AuthenticationError = _FakeAuthenticationError
    mod.RateLimitError = _FakeRateLimitError
    mod.APIConnectionError = _FakeAPIConnectionError
    mod.APIStatusError = _FakeAPIStatusError
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return mod


def _anthropic_provider():
    from dj_engine.enrichment.llm_providers.anthropic_provider import AnthropicProvider
    return AnthropicProvider({})


def test_anthropic_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = _anthropic_provider()
    with pytest.raises(LLMProviderError, match="Geen API-key"):
        provider.complete("prompt", {"api_key_env_var": "ANTHROPIC_API_KEY"})


def test_anthropic_missing_package_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import anthropic -> ImportError
    provider = _anthropic_provider()
    with pytest.raises(LLMProviderError, match="niet geïnstalleerd"):
        provider.complete("prompt", {})


def test_anthropic_success_with_search(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def create_fn(**kwargs):
        return _FakeResponse(content=[
            _FakeBlock("web_search_tool_result"),
            _FakeBlock("text", text='[{"track_id": 1}]'),
        ])

    _install_fake_anthropic(monkeypatch, create_fn)
    provider = _anthropic_provider()

    result = provider.complete("prompt", {"web_search": True, "model": "claude-sonnet-5"})

    assert result.used_search is True
    assert result.text == '[{"track_id": 1}]'
    assert result.model == "claude-sonnet-5"


def test_anthropic_success_without_search_omits_tools(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured = {}

    def create_fn(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(content=[_FakeBlock("text", text="[]")])

    _install_fake_anthropic(monkeypatch, create_fn)
    provider = _anthropic_provider()

    result = provider.complete("prompt", {"web_search": False})

    assert result.used_search is False
    assert "tools" not in captured


def test_anthropic_refusal_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _install_fake_anthropic(
        monkeypatch, lambda **kw: _FakeResponse(content=[], stop_reason="refusal")
    )
    provider = _anthropic_provider()
    with pytest.raises(LLMProviderError, match="weigerde"):
        provider.complete("prompt", {})


def test_anthropic_authentication_error_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-bad")

    def create_fn(**kwargs):
        raise _FakeAuthenticationError("invalid key")

    _install_fake_anthropic(monkeypatch, create_fn)
    provider = _anthropic_provider()
    with pytest.raises(LLMProviderError, match="Ongeldige Anthropic API-key"):
        provider.complete("prompt", {})


def test_anthropic_rate_limit_error_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def create_fn(**kwargs):
        raise _FakeRateLimitError("too many requests")

    _install_fake_anthropic(monkeypatch, create_fn)
    provider = _anthropic_provider()
    with pytest.raises(LLMProviderError, match="rate limit"):
        provider.complete("prompt", {})


def test_anthropic_api_status_error_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def create_fn(**kwargs):
        raise _FakeAPIStatusError("server error", status_code=503)

    _install_fake_anthropic(monkeypatch, create_fn)
    provider = _anthropic_provider()
    with pytest.raises(LLMProviderError, match="503"):
        provider.complete("prompt", {})


def test_anthropic_uses_custom_api_key_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MY_CUSTOM_KEY", "sk-custom")
    _install_fake_anthropic(
        monkeypatch, lambda **kw: _FakeResponse(content=[_FakeBlock("text", text="[]")])
    )
    provider = _anthropic_provider()
    result = provider.complete("prompt", {"api_key_env_var": "MY_CUSTOM_KEY"})
    assert result.text == "[]"


# --- OpenAICompatibleProvider --------------------------------------------

def _openai_provider():
    return OpenAICompatibleProvider({})


def _base_cfg(**overrides):
    cfg = {
        "base_url": "https://api.example.com/v1",
        "model": "gpt-test",
        "api_key_env_var": "TEST_OPENAI_KEY",
    }
    cfg.update(overrides)
    return cfg


def test_openai_compatible_missing_base_url_raises(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    provider = _openai_provider()
    with pytest.raises(LLMProviderError, match="base_url"):
        provider.complete("prompt", _base_cfg(base_url=None))


def test_openai_compatible_missing_model_raises(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    provider = _openai_provider()
    with pytest.raises(LLMProviderError, match="model"):
        provider.complete("prompt", _base_cfg(model=None))


def test_openai_compatible_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
    provider = _openai_provider()
    with pytest.raises(LLMProviderError, match="Geen API-key"):
        provider.complete("prompt", _base_cfg())


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_openai_compatible_success(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")

    def fake_urlopen(request, timeout=None):
        assert request.full_url == "https://api.example.com/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer sk-test"
        return _FakeHTTPResponse({"choices": [{"message": {"content": '[{"track_id": 2}]'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = _openai_provider()

    result = provider.complete("prompt", _base_cfg())

    assert result.text == '[{"track_id": 2}]'
    assert result.used_search is False
    assert result.model == "gpt-test"


def test_openai_compatible_strips_trailing_slash_from_base_url(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeHTTPResponse({"choices": [{"message": {"content": "[]"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = _openai_provider()
    provider.complete("prompt", _base_cfg(base_url="https://api.example.com/v1/"))

    assert captured["url"] == "https://api.example.com/v1/chat/completions"


def test_openai_compatible_http_error_raises(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", hdrs=None, fp=None
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = _openai_provider()
    with pytest.raises(LLMProviderError, match="401"):
        provider.complete("prompt", _base_cfg())


def test_openai_compatible_connection_error_raises(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = _openai_provider()
    with pytest.raises(LLMProviderError, match="Netwerkfout"):
        provider.complete("prompt", _base_cfg())


def test_openai_compatible_malformed_json_raises(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")

    class BadResponse:
        def read(self):
            return b"not json"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: BadResponse())
    provider = _openai_provider()
    with pytest.raises(LLMProviderError, match="niet-JSON"):
        provider.complete("prompt", _base_cfg())


def test_openai_compatible_unexpected_shape_raises(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeHTTPResponse({"unexpected": "shape"}),
    )
    provider = _openai_provider()
    with pytest.raises(LLMProviderError, match="Onverwacht antwoordformaat"):
        provider.complete("prompt", _base_cfg())
