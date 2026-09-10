"""Generieke provider voor elke OpenAI-compatibele chat-completions-API:
OpenAI zelf, Groq, Mistral, DeepSeek, een lokale Ollama/LM Studio-server,
enzovoort — alles wat POST {base_url}/chat/completions in het standaard
OpenAI-berichtformaat begrijpt.

Gebruikt bewust alleen de standaardbibliotheek (urllib), geen extra
dependency: dit is precies de "koppel je eigen API"-optie, en die moet
zonder SDK-aannames werken.

Geen ingebouwde websearch — dat is een provider-specifieke Anthropic-
functie (zie anthropic_provider.py). Deze provider antwoordt dus altijd
op basis van het eigen getrainde model-geheugen; `used_search` staat
daarom altijd op False. Dat is een bewuste, transparante beperking: de
gebruiker ziet in de output altijd of websearch daadwerkelijk gebruikt is.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from dj_engine.enrichment.llm_providers.base import LLMProviderError, LLMResponse

DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 60


class OpenAICompatibleProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def complete(self, prompt: str, config: dict[str, Any] | None = None) -> LLMResponse:
        cfg = config or self.config

        base_url = cfg.get("base_url")
        if not base_url:
            raise LLMProviderError(
                "llm_suggest.base_url ontbreekt in de config — verplicht voor "
                "provider 'openai_compatible' (bv. https://api.openai.com/v1)."
            )
        model = cfg.get("model")
        if not model:
            raise LLMProviderError("llm_suggest.model ontbreekt in de config.")

        api_key_var = cfg.get("api_key_env_var", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_var)
        if not api_key:
            raise LLMProviderError(
                f"Geen API-key gevonden in environment-variabele '{api_key_var}'. "
                f"Zet 'm met: dj-engine set-api-key <key> --var-name {api_key_var}"
            )

        max_tokens = cfg.get("max_output_tokens", DEFAULT_MAX_TOKENS)
        timeout = cfg.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)

        url = base_url.rstrip("/") + "/chat/completions"
        body = json.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise LLMProviderError(f"API-fout ({exc.code}) van {base_url}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise LLMProviderError(f"Netwerkfout bij {base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMProviderError(f"Timeout ({timeout}s) bij {base_url}") from exc
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"Onverwachte (niet-JSON) respons van {base_url}: {exc}") from exc

        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                f"Onverwacht antwoordformaat van {base_url} (geen choices[0].message.content): {exc}"
            ) from exc

        return LLMResponse(text=text, used_search=False, model=model)
