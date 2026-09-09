"""Centrale logging-configuratie. Wordt door de CLI aangeroepen vóórdat
enige module gaat loggen, zodat fouten per bestand (ingest) altijd naar
zowel console als het configureerbare logbestand gaan.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def configure_logging(config: dict[str, Any]) -> None:
    """Configureer root-logging op basis van config['logging'].

    - `level`: bv. "INFO", "DEBUG" (config.yaml -> logging.level)
    - `file`: pad naar logbestand; map wordt aangemaakt indien nodig.
      Leeg/None -> alleen console-logging.
    """
    log_cfg = config.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_file = log_cfg.get("file")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
        force=True,  # herconfigureerbaar (bv. tussen CLI-aanroepen in tests)
    )
