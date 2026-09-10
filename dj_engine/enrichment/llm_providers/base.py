"""Provider-onafhankelijke interface voor de AI-suggestiebron.

Elke provider (anthropic_provider.py, openai_compatible_provider.py, of een
eigen provider die een gebruiker toevoegt) implementeert deze ene methode:
een prompt in, platte tekst + of er websearch gebruikt is terug. Alle
prompt-opbouw en antwoord-parsing (het "vaste algoritme") zit in
recommend/llm_suggest.py, niet hier — dat is precies wat het antwoord
voorspelbaar houdt ongeacht welke provider gekozen is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LLMProviderError(Exception):
    """Nette, gebruikersvriendelijke fout (ontbrekende key, netwerkfout,
    onbekende provider, ...) — vangt de caller (CLI) altijd af zonder crash."""


@dataclass
class LLMResponse:
    text: str
    used_search: bool
    model: str


class LLMProvider(Protocol):
    """Interface die elke provider-adapter implementeert."""

    def complete(self, prompt: str, config: dict[str, Any]) -> LLMResponse:
        """Stuur `prompt` naar het model, retourneer het platte antwoord.

        Raises:
            LLMProviderError: bij elke fout (ontbrekende/ongeldige API-key,
                netwerkfout, timeout, onverwachte API-respons) — nooit een
                provider-specifieke exception laten weglekken naar de caller.
        """
        ...
