"""Tests voor dj_engine.logging_config."""
from __future__ import annotations

import logging

from dj_engine.logging_config import configure_logging


def test_configure_logging_sets_level_and_creates_log_file(tmp_path):
    log_file = tmp_path / "nested" / "dj_engine.log"
    configure_logging({"logging": {"level": "DEBUG", "file": str(log_file)}})

    assert logging.getLogger().level == logging.DEBUG
    logging.getLogger("test").info("hello")
    assert log_file.exists()


def test_configure_logging_without_file_only_uses_console(tmp_path):
    configure_logging({"logging": {"level": "WARNING", "file": None}})
    assert logging.getLogger().level == logging.WARNING
    assert all(not isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers)


def test_configure_logging_defaults_to_info_level():
    configure_logging({"logging": {}})
    assert logging.getLogger().level == logging.INFO
