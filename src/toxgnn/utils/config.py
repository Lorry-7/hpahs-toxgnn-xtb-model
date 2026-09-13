"""Configuration loading, saving, and CLI argument merging utilities."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dict to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base config. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_cli_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI arguments into config. Non-None args override config values."""
    result = copy.deepcopy(config)
    for key, value in vars(args).items():
        if value is not None:
            # Convert snake_case key to nested config path if needed
            if "_" in key:
                parts = key.split("_")
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                result[key] = value
    return result


def resolve_config_path(config_path: str | Path, project_root: str | Path = ".") -> Path:
    """Resolve config path relative to project root if not absolute."""
    config_path = Path(config_path)
    if config_path.is_absolute():
        return config_path
    return Path(project_root) / config_path


def save_resolved_config(
    config: dict[str, Any],
    output_dir: str | Path,
    script_name: str = "unknown",
) -> Path:
    """Save the fully resolved config with metadata to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "_metadata": {
            "script": script_name,
            "timestamp": datetime.now().isoformat(),
            "config_version": "0.2.0",
        },
        **config,
    }

    config_path = output_dir / "config_resolved.yaml"
    save_yaml(resolved, config_path)
    return config_path


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add standard --config argument to argument parser."""
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )
    return parser


def load_and_merge_config(
    config_path: str | Path,
    cli_args: argparse.Namespace | None = None,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Load config from YAML, merge CLI args, and return final config."""
    full_path = resolve_config_path(config_path, project_root)
    config = load_yaml(full_path)

    # Add project_root to config
    config["_project_root"] = str(Path(project_root).resolve())
    config["_config_path"] = str(full_path.resolve())

    if cli_args is not None:
        config = merge_cli_args(config, cli_args)

    return config
