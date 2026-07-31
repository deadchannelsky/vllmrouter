"""Configuration schema and loader for vllm-proxy.

Defines three dataclasses:
  - PoolConfig  — warm-pool settings
  - ModelConfig — per-model VLLM parameters
  - ProxyConfig — top-level config (composes Pool + Models)

`load_config(path)` reads a YAML file and returns a validated ProxyConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import yaml


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
    env: dict[str, str] = field(default_factory=dict)  # extra env vars for this model's subprocess


@dataclass
class ProxyConfig:
    host: str
    port: int
    log_dir: str
    pool: PoolConfig
    models: dict[str, ModelConfig]


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
        env={str(k): str(v) for k, v in raw.get("env", {}).items()},
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

    return ProxyConfig(
        host=str(raw.get("host", "0.0.0.0")),
        port=int(raw.get("port", 8000)),
        log_dir=str(raw.get("log_dir", "logs")),
        pool=pool,
        models=models,
    )
