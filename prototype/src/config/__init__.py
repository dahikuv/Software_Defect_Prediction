"""Configuration helpers for the prototype project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIG_PATH


def load_project_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the project YAML configuration file."""
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config_section(section: str, path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Return one top-level configuration section."""
    return load_project_config(path).get(section, {})

