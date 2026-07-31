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
    env: dict[str, str] = field(default_factory=dict)  # extra env vars for this model's subprocess


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

# vllm's --host defaults to unset, which binds the subprocess's OpenAI-
# compatible port to all interfaces instead of loopback — bypassing this
# proxy's API-key auth entirely. Every hand-configured model in this repo's
# deployments pins --host=127.0.0.1 explicitly; auto-discovered models must
# get the same default or they're a silent, unauthenticated network exposure
# the moment they're cold-started.
_SCAN_DEFAULT_VLLM_ARGS = ["--host=127.0.0.1"]


def _hf_cache_repo_id(dirname: str) -> str | None:
    """Return the ``org/name`` repo id encoded in a HF hub cache dirname.

    The Hugging Face hub cache names each repo's directory
    ``models--{org}--{name}`` (``/`` is replaced with ``--`` when the repo id
    is encoded). Returns ``None`` if *dirname* doesn't match that pattern.
    """
    if not dirname.startswith("models--"):
        return None
    parts = dirname[len("models--") :].split("--")
    if len(parts) != 2 or not all(parts):
        return None
    return "/".join(parts)


def _hf_cache_snapshot_complete(entry_path: str) -> bool:
    """True if *entry_path* (a ``models--org--name`` dir) has a full snapshot.

    Guards against registering a repo whose download was interrupted: a
    complete cache entry has a non-empty ``snapshots/<revision>/`` directory
    (the resolved file tree vllm/transformers actually reads).
    """
    snapshots_dir = os.path.join(entry_path, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return False
    return any(
        os.scandir(revision.path)
        for revision in os.scandir(snapshots_dir)
        if revision.is_dir()
    )


def _slugify_model_id(repo_id: str, taken: set[str]) -> str:
    """Derive a config-friendly model_id from a HF ``org/name`` repo id.

    Uses the repo name (portion after ``/``), lowercased. On collision with an
    id already in *taken*, falls back to a lowercased ``org-name`` form.
    """
    org, name = repo_id.split("/", 1)
    slug = name.lower()
    if slug in taken:
        slug = f"{org.lower()}-{name.lower()}"
    return slug


def scan_and_register_models(config: ProxyConfig, config_path: str) -> list[str]:
    """Scan ``config.model_scan_paths`` and register newly discovered models.

    Two kinds of entries are recognized in each scan path:

    - Hugging Face hub cache repos (dirs named ``models--{org}--{name}``,
      e.g. ``~/.cache/huggingface/hub``): registered with ``model_path`` set
      to the resolved ``org/name`` repo id (not the cache filesystem path),
      matching how vllm/transformers resolve models from the HF cache.
      Incomplete downloads (no snapshot yet) are skipped. Dedup is by
      resolved repo id against existing ``model_path`` values, since a repo
      may already be registered under a hand-chosen model_id.
    - Plain local directories (e.g. a finetunes folder): registered as
      before, with ``model_id`` equal to the directory name and
      ``model_path`` the absolute path to that directory. Dedup is by
      model_id.

    The discovered models are inserted into ``config.models`` in-place **and**
    written back to the YAML file at *config_path* so they survive the next
    ``load_config`` call.

    Returns the list of newly registered model_ids.
    """
    if not config.model_scan_paths:
        return []

    new_model_ids: list[str] = []
    known_model_paths = {cfg.model_path for cfg in config.models.values()}

    for scan_path in config.model_scan_paths:
        if not os.path.isdir(scan_path):
            logger.warning("model_scan_paths entry '%s' is not a directory — skipping", scan_path)
            continue

        for entry in sorted(os.scandir(scan_path), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                # Housekeeping dirs (HF cache's .locks, .no_exist, stray .git, …)
                continue

            repo_id = _hf_cache_repo_id(entry.name)
            if repo_id is not None:
                if not _hf_cache_snapshot_complete(entry.path):
                    logger.warning(
                        "Scan: '%s' looks like an incomplete HF cache download — skipping",
                        entry.name,
                    )
                    continue
                if repo_id in known_model_paths:
                    logger.debug("Scan: HF repo '%s' already registered — skipping", repo_id)
                    continue
                model_id = _slugify_model_id(repo_id, set(config.models))
                config.models[model_id] = ModelConfig(model_path=repo_id, vllm_args=list(_SCAN_DEFAULT_VLLM_ARGS))
                known_model_paths.add(repo_id)
                new_model_ids.append(model_id)
                logger.info("Scan: registered new model '%s' from HF cache repo '%s'", model_id, repo_id)
                continue

            model_id = entry.name
            if model_id in config.models:
                logger.debug("Scan: model '%s' already registered — skipping", model_id)
                continue
            model_path = os.path.abspath(entry.path)
            config.models[model_id] = ModelConfig(model_path=model_path, vllm_args=list(_SCAN_DEFAULT_VLLM_ARGS))
            known_model_paths.add(model_path)
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
            **({"env": cfg.env} if cfg.env else {}),
        }
        for model_id, cfg in config.models.items()
    }

    try:
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.dump(raw, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("Config persisted to '%s' (%d models)", config_path, len(config.models))
    except OSError as exc:
        logger.error("scan_and_register_models: could not write '%s': %s", config_path, exc)
