"""Tests for vllm_proxy.config — load_config validation."""

from __future__ import annotations

import textwrap
import pytest

from vllm_proxy.config import load_config, ProxyConfig, PoolConfig, ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(tmp_path, content: str):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


# ---------------------------------------------------------------------------
# Valid config
# ---------------------------------------------------------------------------


def test_load_valid_config(tmp_path):
    path = write_yaml(
        tmp_path,
        """
        host: "127.0.0.1"
        port: 9999
        log_dir: "logs"
        pool:
          max_size: 3
          base_port: 9000
          startup_timeout_seconds: 60
        models:
          my-model:
            model_path: "/models/my-model"
            vllm_args:
              - "--dtype=float16"
          other-model:
            model_path: "/models/other"
        """,
    )
    cfg = load_config(path)

    assert isinstance(cfg, ProxyConfig)
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9999
    assert cfg.log_dir == "logs"

    assert isinstance(cfg.pool, PoolConfig)
    assert cfg.pool.max_size == 3
    assert cfg.pool.base_port == 9000
    assert cfg.pool.startup_timeout_seconds == 60

    assert "my-model" in cfg.models
    assert "other-model" in cfg.models

    m = cfg.models["my-model"]
    assert isinstance(m, ModelConfig)
    assert m.model_path == "/models/my-model"
    assert m.vllm_args == ["--dtype=float16"]
    assert m.priority == 0


def test_default_values_applied(tmp_path):
    """host, port, log_dir, startup_timeout_seconds should have defaults."""
    path = write_yaml(
        tmp_path,
        """
        pool:
          max_size: 1
          base_port: 9000
        models:
          m:
            model_path: "/m"
        """,
    )
    cfg = load_config(path)
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.log_dir == "logs"
    assert cfg.pool.startup_timeout_seconds == 120


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------


def test_missing_pool_section(tmp_path):
    path = write_yaml(tmp_path, "models:\n  m:\n    model_path: /m\n")
    with pytest.raises(ValueError, match="pool"):
        load_config(path)


def test_missing_models_section(tmp_path):
    path = write_yaml(
        tmp_path,
        "pool:\n  max_size: 1\n  base_port: 9000\n",
    )
    with pytest.raises(ValueError, match="models"):
        load_config(path)


def test_missing_pool_max_size(tmp_path):
    path = write_yaml(
        tmp_path,
        """
        pool:
          base_port: 9000
        models:
          m:
            model_path: /m
        """,
    )
    with pytest.raises(ValueError, match="max_size"):
        load_config(path)


def test_missing_model_path(tmp_path):
    path = write_yaml(
        tmp_path,
        """
        pool:
          max_size: 1
          base_port: 9000
        models:
          broken-model:
            vllm_args: []
        """,
    )
    with pytest.raises(ValueError, match="model_path"):
        load_config(path)


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")
