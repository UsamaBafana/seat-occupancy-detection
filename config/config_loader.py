"""
config/config_loader.py
Loads and validates YAML configuration with dot-access support.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Recursive dot-access wrapper around a dict loaded from YAML."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, Config) else value
        return result

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"


def load_config(path: str | Path | None = None) -> Config:
    """Load settings.yaml and return a Config object."""
    if path is None:
        path = Path(__file__).parent / "settings.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(raw)


def override_config(cfg: Config, overrides: dict) -> Config:
    """
    Apply flat key=value overrides to a Config.
    Keys use dot notation, e.g. {'model.confidence_threshold': 0.5}
    """
    data = cfg.to_dict()
    for dotkey, val in overrides.items():
        parts = dotkey.split(".")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return Config(data)
