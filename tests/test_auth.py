"""Tests for vllm_proxy.auth — load_keys() and the auth middleware."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllm_proxy.auth import load_keys
from vllm_proxy.config import ModelConfig, PoolConfig, ProxyConfig
from vllm_proxy.app import create_app
from vllm_proxy.pool import ModelPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_KEY = "sk-test-validkey"
OTHER_KEY = "sk-test-otherkey"


def make_config() -> ProxyConfig:
    return ProxyConfig(
        host="127.0.0.1",
        port=8000,
        log_dir="/tmp/test-logs",
        pool=PoolConfig(max_size=2, base_port=9000),
        models={
            "model-a": ModelConfig(model_path="/models/a"),
        },
    )


def _make_app(keys_file: str):
    """Create app with VLLM_KEYS_FILE set and ModelPool patched."""
    fake_pool = MagicMock(spec=ModelPool)
    fake_pool.list_loaded = MagicMock(return_value=[])
    fake_pool.get_or_load = AsyncMock()
    fake_pool.unload = AsyncMock()
    fake_pool.shutdown_all = AsyncMock()

    with patch.dict(os.environ, {"VLLM_KEYS_FILE": keys_file}):
        with patch("vllm_proxy.app.ModelPool", return_value=fake_pool):
            app = create_app(make_config())

    return app


# ---------------------------------------------------------------------------
# load_keys() unit tests
# ---------------------------------------------------------------------------


def test_load_keys_returns_keys(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("sk-one\nsk-two\n")
    result = load_keys(str(key_file))
    assert result == frozenset({"sk-one", "sk-two"})


def test_load_keys_skips_blank_lines(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("\n  \nsk-real\n\n")
    result = load_keys(str(key_file))
    assert result == frozenset({"sk-real"})


def test_load_keys_skips_comments(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("# this is a comment\nsk-actual\n# another comment\n")
    result = load_keys(str(key_file))
    assert result == frozenset({"sk-actual"})


def test_load_keys_empty_file_returns_empty_frozenset(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("")
    result = load_keys(str(key_file))
    assert result == frozenset()


def test_load_keys_strips_whitespace(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("  sk-padded  \n")
    result = load_keys(str(key_file))
    assert "sk-padded" in result


def test_load_keys_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_keys(str(tmp_path / "nonexistent.txt"))


# ---------------------------------------------------------------------------
# Middleware integration tests
# ---------------------------------------------------------------------------


def test_valid_key_passes_through(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text(f"{VALID_KEY}\n")
    app = _make_app(str(key_file))

    with TestClient(app) as client:
        resp = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
    assert resp.status_code == 200


def test_missing_auth_header_returns_401(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text(f"{VALID_KEY}\n")
    app = _make_app(str(key_file))

    with TestClient(app) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 401


def test_malformed_auth_header_returns_401(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text(f"{VALID_KEY}\n")
    app = _make_app(str(key_file))

    with TestClient(app) as client:
        resp = client.get(
            "/v1/models",
            headers={"Authorization": VALID_KEY},  # missing "Bearer " prefix
        )
    assert resp.status_code == 401


def test_wrong_key_returns_403(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text(f"{VALID_KEY}\n")
    app = _make_app(str(key_file))

    with TestClient(app) as client:
        resp = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {OTHER_KEY}"},
        )
    assert resp.status_code == 403


def test_unreadable_key_file_returns_503(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text(f"{VALID_KEY}\n")
    app = _make_app(str(key_file))

    # After the app is built, patch load_keys to simulate an unreadable file.
    with patch("vllm_proxy.app.load_keys", side_effect=OSError("disk error")):
        with TestClient(app) as client:
            resp = client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {VALID_KEY}"},
            )
    assert resp.status_code == 503


def test_create_app_raises_without_keys_file_env_var():
    env = {k: v for k, v in os.environ.items() if k != "VLLM_KEYS_FILE"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="VLLM_KEYS_FILE"):
            create_app(make_config())


def test_admin_route_also_requires_auth(tmp_path: Path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text(f"{VALID_KEY}\n")
    app = _make_app(str(key_file))

    with TestClient(app) as client:
        # No auth header
        resp = client.get("/admin/models")
    assert resp.status_code == 401

    with TestClient(app) as client:
        # Valid auth header
        resp = client.get(
            "/admin/models",
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
    assert resp.status_code == 200
