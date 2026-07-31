"""Configuration schema and loader for vllm-proxy.

Defines three dataclasses:
  - PoolConfig  — warm-pool settings
  - ModelConfig — per-model VLLM parameters
  - ProxyConfig — top-level config (composes Pool + Models)

`load_config(path)` reads a YAML file and returns a validated ProxyConfig.
`scan_and_register_models(config, config_path)` scans each directory in
``config.model_scan_paths`` and registers any new model sub-directories.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PoolConfig:
    max_size: int
    base_port: int
    startup_timeout_seconds: int = 120


@dataclass
class ModelConfig:
    model_path: str
    vllm_args: list[str] = field(default_factory=list)
    priority: int = 0  # reserved — not used yet


@dataclass
class ProxyConfig:
    host: str
    port: int
    log_dir: str
    pool: PoolConfig
    models: dict[str, ModelConfig]
    model_scan_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_pool(raw: dict[str, Any]) -> PoolConfig:
    missing = [k for k in ("max_size", "base_port") if k not in raw]
    if missing:
        raise ValueError(f"pool config missing required fields: {missing}")
    return PoolConfig(
        max_size=int(raw["max_size"]),
        base_port=int(raw["base_port"]),
        startup_timeout_seconds=int(raw.get("startup_timeout_seconds", 120)),
    )


def _parse_model(model_id: str, raw: dict[str, Any]) -> ModelConfig:
    if "model_path" not in raw:
        raise ValueError(f"model '{model_id}' is missing required field 'model_path'")
    return ModelConfig(
        model_path=raw["model_path"],
        vllm_args=list(raw.get("vllm_args", [])),
        priority=int(raw.get("priority", 0)),
    )


def load_config(path: str) -> ProxyConfig:
    """Read *path* as YAML and return a validated :class:`ProxyConfig`."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got: {type(raw).__name__}")

    missing_top = [k for k in ("pool", "models") if k not in raw]
    if missing_top:
        raise ValueError(f"Config missing required top-level keys: {missing_top}")

    pool = _parse_pool(raw["pool"])

    raw_models = raw["models"]
    if not isinstance(raw_models, dict):
        raise ValueError("'models' must be a YAML mapping of model_id → model config")

    models: dict[str, ModelConfig] = {
        model_id: _parse_model(model_id, cfg or {})
        for model_id, cfg in raw_models.items()
    }

    scan_paths_raw = raw.get("model_scan_paths", [])
    if not isinstance(scan_paths_raw, list):
        raise ValueError("'model_scan_paths' must be a YAML sequence of paths")
    model_scan_paths = [str(p) for p in scan_paths_raw]

    return ProxyConfig(
        host=str(raw.get("host", "0.0.0.0")),
        port=int(raw.get("port", 8000)),
        log_dir=str(raw.get("log_dir", "logs")),
        pool=pool,
        models=models,
        model_scan_paths=model_scan_paths,
    )


# ---------------------------------------------------------------------------
# Model directory scanner
# ---------------------------------------------------------------------------


def scan_and_register_models(config: ProxyConfig, config_path: str) -> list[str]:
    """Scan ``config.model_scan_paths`` and register newly discovered models.

    For each scan path, every immediate sub-directory is treated as a potential
    model whose ``model_id`` equals the directory name and whose
    ``model_path`` is the full absolute path to that directory.  Only entries
    not already present in ``config.models`` are added.

    The discovered models are inserted into ``config.models`` in-place **and**
    written back to the YAML file at *config_path* so they survive the next
    ``load_config`` call.

    Returns the list of newly registered model_ids.
    """
    if not config.model_scan_paths:
        return []

    new_model_ids: list[str] = []

    for scan_path in config.model_scan_paths:
        if not os.path.isdir(scan_path):
            logger.warning("model_scan_paths entry '%s' is not a directory — skipping", scan_path)
            continue

        for entry in sorted(os.scandir(scan_path), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            model_id = entry.name
            if model_id in config.models:
                logger.debug("Scan: model '%s' already registered — skipping", model_id)
                continue
            model_path = os.path.abspath(entry.path)
            config.models[model_id] = ModelConfig(model_path=model_path)
            new_model_ids.append(model_id)
            logger.info("Scan: registered new model '%s' from '%s'", model_id, model_path)

    if new_model_ids:
        _persist_models(config, config_path)

    return new_model_ids


def _persist_models(config: ProxyConfig, config_path: str) -> None:
    """Write the current ``config.models`` mapping back into the YAML file.

    Only the ``models`` key is updated; all other keys and comments in the
    file are preserved via a round-trip read → merge → write.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except OSError as exc:
        logger.error("scan_and_register_models: could not read '%s' for update: %s", config_path, exc)
        return

    raw["models"] = {
        model_id: {
            "model_path": cfg.model_path,
            **({"vllm_args": cfg.vllm_args} if cfg.vllm_args else {}),
            **({"priority": cfg.priority} if cfg.priority != 0 else {}),
        }
        for model_id, cfg in config.models.items()
    }

    try:
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.dump(raw, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("Config persisted to '%s' (%d models)", config_path, len(config.models))
    except OSError as exc:
        logger.error("scan_and_register_models: could not write '%s': %s", config_path, exc)
