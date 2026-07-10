from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _parse_override(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return parsed


def _set_nested(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("Config file must load into a dictionary.")
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    merged = deepcopy(config)
    if not overrides:
        return merged

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value format.")
        key, value = item.split("=", 1)
        _set_nested(merged, key.strip(), _parse_override(value.strip()))
    return merged


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    return apply_overrides(config, overrides)


def load_merged_config(default_config_path: str | Path, user_config_path: str | Path | None = None) -> dict[str, Any]:
    default_config = load_yaml_config(default_config_path)
    if user_config_path is None:
        return default_config
    user_config = load_yaml_config(user_config_path)
    return deep_merge_dicts(default_config, user_config)
