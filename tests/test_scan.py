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


def test_discovered_plain_dir_model_defaults_to_loopback_host(tmp_path):
    """vllm's --host defaults to binding all interfaces, not loopback — an
    auto-discovered model must not be silently network-exposed."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "llama3-8b").mkdir()

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    scan_and_register_models(cfg, path)

    assert cfg.models["llama3-8b"].vllm_args == ["--host=127.0.0.1"]


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


# ---------------------------------------------------------------------------
# Hugging Face hub cache layout
# ---------------------------------------------------------------------------


def make_hf_repo(cache_dir, org: str, name: str, complete: bool = True):
    """Create a ``models--org--name`` dir mimicking the HF hub cache layout."""
    repo_dir = cache_dir / f"models--{org}--{name}"
    repo_dir.mkdir()
    if complete:
        snapshot = repo_dir / "snapshots" / "abcd1234"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
    return repo_dir


def test_hf_cache_repo_registered_with_repo_id_as_model_path(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    make_hf_repo(cache, "google", "gemma-4-31B-it")

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    result = scan_and_register_models(cfg, path)

    assert result == ["gemma-4-31b-it"]
    assert cfg.models["gemma-4-31b-it"].model_path == "google/gemma-4-31B-it"


def test_hf_cache_repo_defaults_to_loopback_host(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    make_hf_repo(cache, "google", "gemma-4-31B-it")

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    scan_and_register_models(cfg, path)

    assert cfg.models["gemma-4-31b-it"].vllm_args == ["--host=127.0.0.1"]


def test_hf_cache_repo_already_registered_is_skipped_regardless_of_model_id(tmp_path):
    """A repo already configured under a hand-picked id must not be re-added."""
    cache = tmp_path / "hub"
    cache.mkdir()
    make_hf_repo(cache, "Qwen", "Qwen3.6-27B-FP8")

    cfg, path = make_config(
        tmp_path,
        scan_paths=[str(cache)],
        extra_models={"qwen3.6-27b-fp8": "Qwen/Qwen3.6-27B-FP8"},
    )
    result = scan_and_register_models(cfg, path)

    assert result == []
    assert list(cfg.models) == ["qwen3.6-27b-fp8"]


def test_hf_cache_incomplete_download_is_skipped(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    make_hf_repo(cache, "org", "half-downloaded", complete=False)

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    result = scan_and_register_models(cfg, path)

    assert result == []
    assert "half-downloaded" not in cfg.models


def test_hf_cache_model_id_collision_falls_back_to_org_prefixed_slug(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    make_hf_repo(cache, "some-org", "widget")

    cfg, path = make_config(
        tmp_path,
        scan_paths=[str(cache)],
        extra_models={"widget": "other-org/widget-v0"},
    )
    result = scan_and_register_models(cfg, path)

    assert result == ["some-org-widget"]
    assert cfg.models["some-org-widget"].model_path == "some-org/widget"


def test_hf_cache_housekeeping_dirs_are_ignored(tmp_path):
    """.locks, .no_exist, etc. living alongside models--... dirs must not be registered."""
    cache = tmp_path / "hub"
    cache.mkdir()
    (cache / ".locks").mkdir()
    (cache / ".no_exist").mkdir()
    make_hf_repo(cache, "google", "gemma-4-31B-it")

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    result = scan_and_register_models(cfg, path)

    assert result == ["gemma-4-31b-it"]
    assert ".locks" not in cfg.models
    assert ".no_exist" not in cfg.models


def test_hf_cache_and_plain_dirs_in_same_scan_path(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    make_hf_repo(cache, "ibm-granite", "granite-4.1-30b")
    (cache / "my-finetune").mkdir()

    cfg, path = make_config(tmp_path, scan_paths=[str(cache)])
    result = scan_and_register_models(cfg, path)

    assert sorted(result) == ["granite-4.1-30b", "my-finetune"]
    assert cfg.models["granite-4.1-30b"].model_path == "ibm-granite/granite-4.1-30b"
    assert cfg.models["my-finetune"].model_path == str(cache / "my-finetune")


def test_persist_preserves_env_field(tmp_path):
    """_persist_models must round-trip the env field, not silently drop it."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "new-model").mkdir()

    cfg, path = make_config(
        tmp_path,
        scan_paths=[str(cache)],
        extra_models={"has-env": "/models/has-env"},
    )
    cfg.models["has-env"].env = {"VLLM_USE_DEEP_GEMM": "0"}

    scan_and_register_models(cfg, path)

    reloaded = load_config(path)
    assert reloaded.models["has-env"].env == {"VLLM_USE_DEEP_GEMM": "0"}
