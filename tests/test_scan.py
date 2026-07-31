"""Tests for scan_and_register_models and _persist_models in vllm_proxy.config."""

from __future__ import annotations

import textwrap
import pytest
import yaml

from vllm_proxy.config import (
    load_config,
    scan_and_register_models,
    ModelConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


def make_config(tmp_path, scan_paths: list[str] | None = None, extra_models: dict | None = None):
    """Write a minimal config.yaml with optional scan paths and return (cfg, path)."""
    scan_block = ""
    if scan_paths is not None:
        if scan_paths:
            items = "\n".join(f'    - "{p}"' for p in scan_paths)
            scan_block = f"model_scan_paths:\n{items}\n"
        else:
            scan_block = "model_scan_paths: []\n"

    models_block = "models:\n"
    if extra_models:
        for mid, mpath in extra_models.items():
            models_block += f"  {mid}:\n    model_path: {mpath}\n"
    else:
        models_block += "  existing-model:\n    model_path: /models/existing-model\n"

    content = f"""\
host: "0.0.0.0"
port: 8000
log_dir: logs
{scan_block}
pool:
  max_size: 2
  base_port: 9000
{models_block}
"""
    path = write_yaml(tmp_path, content)
    return load_config(path), path


# ---------------------------------------------------------------------------
# scan_and_register_models — no scan paths
# ---------------------------------------------------------------------------


def test_no_scan_paths_returns_empty(tmp_path):
    cfg, path = make_config(tmp_path, scan_paths=[])
    result = scan_and_register_models(cfg, path)
    assert result == []


def test_scan_paths_absent_returns_empty(tmp_path):
    """Config without model_scan_paths key at all should also be a no-op."""
    cfg, path = make_config(tmp_path, scan_paths=None)
    result = scan_and_register_models(cfg, path)
    assert result == []


# ---------------------------------------------------------------------------
# scan_and_register_models — path does not exist
# ---------------------------------------------------------------------------


def test_nonexistent_scan_path_is_skipped(tmp_path):
    cfg, path = make_config(tmp_path, scan_paths=["/nonexistent/path/that/does/not/exist"])
    result = scan_and_register_models(cfg, path)
    assert result == []
    assert "existing-model" in cfg.models  # original models untouched


# ---------------------------------------------------------------------------
# scan_and_register_models — discovers new models
# ---------------------------------------------------------------------------


def test_discovers_new_model_directories(tmp_path):
    # Create a cache dir with two model subdirectories.
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "llama3-8b").mkdir()
    (cache / "mistral-7b").mkdir()

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    result = scan_and_register_models(cfg, path)

    assert sorted(result) == ["llama3-8b", "mistral-7b"]
    assert "llama3-8b" in cfg.models
    assert "mistral-7b" in cfg.models
    assert cfg.models["llama3-8b"].model_path == str(cache / "llama3-8b")
    assert cfg.models["mistral-7b"].model_path == str(cache / "mistral-7b")


def test_skips_files_not_dirs(tmp_path):
    """Files inside the scan path are ignored; only directories become models."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "real-model").mkdir()
    (cache / "readme.txt").write_text("not a model")

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    result = scan_and_register_models(cfg, path)

    assert result == ["real-model"]
    assert "readme.txt" not in cfg.models


def test_skips_already_registered_models(tmp_path):
    """Models whose name is already in config.models are not re-registered."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "existing-model").mkdir()
    (cache / "brand-new-model").mkdir()

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    original_path = cfg.models["existing-model"].model_path

    result = scan_and_register_models(cfg, path)

    assert result == ["brand-new-model"]
    # Original registration must be unchanged.
    assert cfg.models["existing-model"].model_path == original_path


# ---------------------------------------------------------------------------
# Persistence — YAML file is updated
# ---------------------------------------------------------------------------


def test_persists_new_models_to_yaml(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "new-model").mkdir()

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    scan_and_register_models(cfg, path)

    reloaded = load_config(path)
    assert "new-model" in reloaded.models
    assert reloaded.models["new-model"].model_path == str(cache / "new-model")


def test_yaml_retains_existing_fields_after_persist(tmp_path):
    """Non-models keys (host, port, pool, …) survive the persist round-trip."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "extra-model").mkdir()

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    scan_and_register_models(cfg, path)

    reloaded = load_config(path)
    assert reloaded.host == "0.0.0.0"
    assert reloaded.port == 8000
    assert reloaded.pool.max_size == 2
    assert reloaded.pool.base_port == 9000


def test_no_yaml_write_when_nothing_new(tmp_path):
    """If no new models are found, the YAML file must not be rewritten."""
    cache = tmp_path / "cache"
    cache.mkdir()
    # cache directory is empty — nothing to discover

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    mtime_before = (tmp_path / "config.yaml").stat().st_mtime

    scan_and_register_models(cfg, path)

    mtime_after = (tmp_path / "config.yaml").stat().st_mtime
    assert mtime_before == mtime_after


# ---------------------------------------------------------------------------
# Multiple scan paths
# ---------------------------------------------------------------------------


def test_multiple_scan_paths(tmp_path):
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"
    cache_a.mkdir()
    cache_b.mkdir()
    (cache_a / "model-a").mkdir()
    (cache_b / "model-b").mkdir()

    cfg, path = make_config(tmp_path, scan_paths=[str(cache_a), str(cache_b)])
    result = scan_and_register_models(cfg, path)

    assert sorted(result) == ["model-a", "model-b"]
    assert "model-a" in cfg.models
    assert "model-b" in cfg.models
