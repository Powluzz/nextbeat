"""Config loading voor dj-engine.

Alle instelbare parameters (database-locatie, rate limits, modelpaden,
scoringsgewichten) horen in config.yaml, niet hardcoded in de modules.
"""
from __future__ import annotations

import copy
import os
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
    "rekordbox": {
        # Prefix-herschrijfregels voor Rekordbox' opgeslagen paden (vaak
        # Windows, bv. "C:/Users/naam/Music") naar een pad dat op dit
        # systeem daadwerkelijk bestaat. Leeg = paden ongewijzigd laten
        # (import werkt dan alleen op metadata-niveau totdat dit ingevuld is).
        "path_mapping": [],
    },
    "suggest": {
        # Aantal kandidaten dat suggest_next() intern doorgeeft aan de
        # achtergrond-verfijningsstap (refine.refine_candidates()).
        "shortlist_size": 40,
    },
    "analysis_pool": {
        # Aantal parallelle workerprocessen voor refine.refine_candidates().
        # Elke TF-inference gebruikt intern ~2 cores; hou hier rekening mee
        # t.o.v. het aantal beschikbare cores.
        "workers": 4,
    },
    "claude_api": {
        # Infrastructuur voor een optionele, nog niet gebouwde AI-suggestie-
        # bron (--source llm). Nu alleen configuratie; geen functionaliteit.
        "enabled": False,
        "model": "claude-sonnet-5",
        "api_key_env_var": "ANTHROPIC_API_KEY",
    },
    "scoring": {
        "bpm_max_deviation_pct": 8.0,
        "default_exclude_recent": 5,
        # Drempel voor db.maybe_normalize_energy(): pas herschalen na dit
        # aantal 'pending' tracks (energy_raw gezet, energy nog niet), i.p.v.
        # bij elke losse on-demand analyse.
        "energy_renormalize_threshold": 20,
        "camelot_scores": {
            "same": 100,
            "adjacent_same_letter": 90,
            "relative_major_minor": 85,
            "two_step_same_letter": 75,
            "other": 30,
        },
        "energy_targets": {
            "build": {"center": 0.25, "plateau": 0.10, "falloff": 0.50},
            "hold": {"center": 0.0, "plateau": 0.05, "falloff": 0.35},
            "ease": {"center": -0.25, "plateau": 0.10, "falloff": 0.50},
            "surprise": {"center": 0.0, "plateau": 1.0, "falloff": 0.50},
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


def _local_override_path(config_path: Path) -> Path:
    """'config.yaml' -> 'config.local.yaml' (zelfde map)."""
    return config_path.with_name(f"{config_path.stem}.local{config_path.suffix}")


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Laad config.yaml, vul aan met defaults, en merge er een lokale
    override-laag overheen indien aanwezig.

    Precedentie (hoog naar laag): config.local.yaml > config.yaml > defaults.

    De local-override is bedoeld voor machine-/persoonsspecifieke waarden
    die niet in git horen (bv. rekordbox.path_mapping — een absoluut pad
    naar iemands eigen muziekmap). Zie config.local.yaml.example. Het
    bestand heet altijd '<stem-van-path>.local<extensie>' in dezelfde map
    als `path` (default: config.local.yaml naast config.yaml) en staat in
    .gitignore — het wordt hier automatisch meegeladen als het bestaat,
    zonder dat je er zelf naar hoeft te verwijzen.

    Args:
        path: pad naar config.yaml. Default: config.yaml in projectroot.

    Returns:
        Genest dict met configuratie.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    result = copy.deepcopy(_DEFAULTS)
    if config_path.exists():
        result = _deep_merge(result, _read_yaml(config_path))

    local_path = _local_override_path(config_path)
    if local_path.exists():
        result = _deep_merge(result, _read_yaml(local_path))

    return result


def load_dotenv(path: str | Path = ".env") -> None:
    """Laad KEY=VALUE-regels uit een .env-bestand in os.environ.

    Bestaande environment-variabelen hebben altijd voorrang (worden niet
    overschreven) — zo blijft een expliciet in de shell gezette variabele
    leidend. Geen dependency op python-dotenv nodig voor dit ene doel:
    secrets (zoals ANTHROPIC_API_KEY) handmatig kunnen toevoegen zonder ze
    in config.yaml (wél in git) te zetten. Zie ook cli.py's `set-api-key`.

    Stilzwijgend een no-op als het bestand niet bestaat.
    """
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
