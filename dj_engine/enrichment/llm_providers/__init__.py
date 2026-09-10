"""LLM-providers voor de optionele AI-suggestiebron (recommend/llm_suggest.py).

Ontwerpdoel: de gebruiker kan zelf kiezen welke API hij koppelt (niet
alleen Claude) — maar elke provider krijgt exact hetzelfde prompt/
antwoord-contract voorgeschoteld (zie llm_suggest.build_prompt()), zodat
het resultaat zo voorspelbaar mogelijk blijft ongeacht de gekozen API.

Provider-keuze en instellingen komen uit config.yaml -> llm_suggest.
"""
from __future__ import annotations

from typing import Any

from dj_engine.enrichment.llm_providers.base import LLMProvider, LLMProviderError, LLMResponse

_PROVIDER_NAMES = ("anthropic", "openai_compatible")


def get_provider(config: dict[str, Any]) -> LLMProvider:
    """Fabriceer de geconfigureerde LLMProvider.

    Raises:
        LLMProviderError: onbekende provider, ontbrekende API-key, of een
            ontbrekende verplichte instelling (bv. base_url voor
            openai_compatible) — altijd met een duidelijke, Nederlandse
            boodschap, nooit een kale stacktrace.
    """
    llm_config = config.get("llm_suggest", {})
    provider_name = llm_config.get("provider", "anthropic")

    if provider_name == "anthropic":
        from dj_engine.enrichment.llm_providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(llm_config)
    if provider_name == "openai_compatible":
        from dj_engine.enrichment.llm_providers.openai_compatible_provider import (
            OpenAICompatibleProvider,
        )
        return OpenAICompatibleProvider(llm_config)

    raise LLMProviderError(
        f"Onbekende provider '{provider_name}' (config: llm_suggest.provider) — "
        f"verwacht één van: {', '.join(_PROVIDER_NAMES)}"
    )


__all__ = ["LLMProvider", "LLMProviderError", "LLMResponse", "get_provider"]
