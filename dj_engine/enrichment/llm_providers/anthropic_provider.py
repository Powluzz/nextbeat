"""Anthropic (Claude) provider — enige provider hier met ingebouwde,
server-side websearch (web_search-tool), wat 'm geschikter maakt om
daadwerkelijk te checken of een kandidaat-track in bestaande DJ-tracklists
voorkomt i.p.v. alleen op getraind model-geheugen te varen.
"""
from __future__ import annotations

import os
from typing import Any

from dj_engine.enrichment.llm_providers.base import LLMProviderError, LLMResponse

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4096
# Aparte, gedocumenteerde begrenzing op het aantal websearch-aanroepen per
# suggestie-call — voorkomt onbeperkte (en dus onvoorspelbaar dure) search-
# loops. Zie config: llm_suggest.max_search_uses.
DEFAULT_MAX_SEARCH_USES = 5


class AnthropicProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def complete(self, prompt: str, config: dict[str, Any] | None = None) -> LLMResponse:
        cfg = config or self.config
        api_key_var = cfg.get("api_key_env_var", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_var)
        if not api_key:
            raise LLMProviderError(
                f"Geen API-key gevonden in environment-variabele '{api_key_var}'. "
                f"Zet 'm met: dj-engine set-api-key <key> --var-name {api_key_var}"
            )

        try:
            import anthropic
        except ImportError as exc:
            raise LLMProviderError(
                f"Het 'anthropic'-pakket is niet geïnstalleerd: {exc}. "
                "Installeer met: pip install anthropic"
            ) from exc

        model = cfg.get("model", DEFAULT_MODEL)
        max_tokens = cfg.get("max_output_tokens", DEFAULT_MAX_TOKENS)
        use_search = cfg.get("web_search", True)

        tools = []
        if use_search:
            max_uses = cfg.get("max_search_uses", DEFAULT_MAX_SEARCH_USES)
            tools.append({"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses})

        client = anthropic.Anthropic(api_key=api_key)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            create_kwargs["tools"] = tools

        try:
            response = client.messages.create(**create_kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMProviderError(f"Ongeldige Anthropic API-key: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMProviderError(f"Anthropic rate limit bereikt: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMProviderError(f"Netwerkfout bij Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMProviderError(f"Anthropic API-fout ({exc.status_code}): {exc.message}") from exc

        if response.stop_reason == "refusal":
            raise LLMProviderError("Anthropic weigerde deze aanvraag (refusal) — geen suggestie beschikbaar.")

        text_parts = []
        used_search = False
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "web_search_tool_result":
                used_search = True

        return LLMResponse(text="\n".join(text_parts), used_search=used_search, model=model)
