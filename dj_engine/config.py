"""Config loading voor dj-engine.

Alle instelbare parameters (database-locatie, rate limits, modelpaden,
scoringsgewichten) horen in config.yaml, niet hardcoded in de modules.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Fallback-defaults, alleen gebruikt als config.yaml geen waarde voor een pad bevat.
_DEFAULTS: dict[str, Any] = {
    "database": {"path": "data/dj_engine.db"},
    "audio": {
        "extensions": [".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a"],
        "sample_rate": 44100,
    },
    "musicbrainz": {
        "rate_limit_per_second": 1.0,
        "user_agent_name": "dj-engine",
        "user_agent_version": "0.1.0",
        "user_agent_contact": "",
        "max_retries": 3,
        "timeout_seconds": 10,
    },
    "models": {
        "directory": "models",
        "embedding_model": "discogs-effnet-bs64-1.pb",
        "mood_models": {},
        "genre_model": "genre_discogs400-discogs-effnet-1.pb",
        "genre_labels": "genre_discogs400-discogs-effnet-1.json",
    },
    "scoring": {
        "bpm_max_deviation_pct": 8.0,
        "default_exclude_recent": 5,
        "camelot_scores": {
            "same": 100,
            "adjacent_same_letter": 90,
            "relative_major_minor": 85,
            "two_step_same_letter": 75,
            "other": 30,
        },
        "weights": {
            "build": {"key": 0.35, "bpm": 0.25, "energy": 0.30, "mood": 0.10},
            "hold": {"key": 0.40, "bpm": 0.35, "energy": 0.15, "mood": 0.10},
            "ease": {"key": 0.35, "bpm": 0.20, "energy": 0.35, "mood": 0.10},
            "surprise": {"key": 0.15, "bpm": 0.15, "energy": 0.20, "mood": 0.50},
        },
    },
    "logging": {"level": "INFO", "file": "logs/dj_engine.log"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override in base (recursief), retourneert nieuwe dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Laad config.yaml en vul ontbrekende waarden aan met defaults.

    Args:
        path: pad naar config.yaml. Default: config.yaml in projectroot.

    Returns:
        Genest dict met configuratie.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return copy.deepcopy(_DEFAULTS)

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    return _deep_merge(_DEFAULTS, user_config)
