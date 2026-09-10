"""Tests voor dj_engine.config."""
from __future__ import annotations

import os

from dj_engine.config import load_config, load_dotenv


def test_load_config_missing_file_returns_defaults(tmp_path):
    config = load_config(tmp_path / "does_not_exist.yaml")
    assert config["database"]["path"] == "data/dj_engine.db"
    assert config["rekordbox"]["path_mapping"] == []
    assert config["analysis_pool"]["workers"] == 4
    assert config["claude_api"]["enabled"] is False
    assert config["scoring"]["energy_renormalize_threshold"] == 20


def test_load_config_user_yaml_overrides_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("rekordbox:\n  path_mapping:\n    - from: 'C:/Music'\n      to: '/mnt/music'\n")

    config = load_config(cfg_file)

    assert config["rekordbox"]["path_mapping"] == [{"from": "C:/Music", "to": "/mnt/music"}]
    # niet-overschreven secties blijven op hun default staan
    assert config["analysis_pool"]["workers"] == 4


def test_load_config_local_override_wins_over_base(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "rekordbox:\n  path_mapping:\n    - from: 'shared'\n      to: 'shared-target'\n"
    )
    (tmp_path / "config.local.yaml").write_text(
        "rekordbox:\n  path_mapping:\n    - from: 'C:/Users/mij/Music'\n      to: '/home/mij/muziek'\n"
    )

    config = load_config(tmp_path / "config.yaml")

    assert config["rekordbox"]["path_mapping"] == [{"from": "C:/Users/mij/Music", "to": "/home/mij/muziek"}]


def test_load_config_local_override_without_base_file(tmp_path):
    """De local-override moet ook werken als config.yaml zelf niet bestaat
    (dan gelden de ingebouwde defaults + de local-override erbovenop)."""
    (tmp_path / "config.local.yaml").write_text("database:\n  path: 'ergens.db'\n")

    config = load_config(tmp_path / "config.yaml")

    assert config["database"]["path"] == "ergens.db"
    assert config["scoring"]["bpm_max_deviation_pct"] == 8.0  # defaults blijven staan


def test_load_config_no_local_override_file_is_fine(tmp_path):
    (tmp_path / "config.yaml").write_text("database:\n  path: 'x.db'\n")
    config = load_config(tmp_path / "config.yaml")
    assert config["database"]["path"] == "x.db"


def test_load_config_partial_override_keeps_sibling_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("scoring:\n  energy_renormalize_threshold: 5\n")

    config = load_config(cfg_file)

    assert config["scoring"]["energy_renormalize_threshold"] == 5
    assert config["scoring"]["bpm_max_deviation_pct"] == 8.0  # ongemoeid


# --- load_dotenv --------------------------------------------------------

def test_load_dotenv_sets_env_vars(tmp_path, monkeypatch):
    monkeypatch.delenv("DJ_ENGINE_TEST_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DJ_ENGINE_TEST_VAR=hello\n")

    load_dotenv(env_file)

    assert os.environ["DJ_ENGINE_TEST_VAR"] == "hello"
    del os.environ["DJ_ENGINE_TEST_VAR"]


def test_load_dotenv_does_not_override_existing_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("DJ_ENGINE_TEST_VAR", "from_shell")
    env_file = tmp_path / ".env"
    env_file.write_text("DJ_ENGINE_TEST_VAR=from_dotenv\n")

    load_dotenv(env_file)

    assert os.environ["DJ_ENGINE_TEST_VAR"] == "from_shell"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    load_dotenv(tmp_path / "does_not_exist.env")  # mag niet crashen


def test_load_dotenv_ignores_comments_and_blank_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("DJ_ENGINE_TEST_VAR2", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\n\nDJ_ENGINE_TEST_VAR2=\"quoted value\"\n")

    load_dotenv(env_file)

    assert os.environ["DJ_ENGINE_TEST_VAR2"] == "quoted value"
    del os.environ["DJ_ENGINE_TEST_VAR2"]
