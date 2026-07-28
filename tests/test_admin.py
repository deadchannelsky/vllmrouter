"""Tests for vllm_proxy.admin — each admin endpoint, ephemeral config mutations."""

from __future__ import annotations

import os
import pathlib
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllm_proxy.config import ModelConfig, PoolConfig, ProxyConfig
from vllm_proxy.app import create_app
from vllm_proxy.pool import ModelPool

# A stable test key injected into every test via the patched key file.
_TEST_KEY = "sk-test-admin"
_AUTH_HEADER = {"Authorization": f"Bearer {_TEST_KEY}"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_config() -> ProxyConfig:
    return ProxyConfig(
        host="127.0.0.1",
        port=8000,
        log_dir="/tmp/test-logs",
        pool=PoolConfig(max_size=2, base_port=9000),
        models={
            "model-a": ModelConfig(model_path="/models/a"),
            "model-b": ModelConfig(model_path="/models/b"),
        },
    )


def _make_app_with_fake_pool(config: ProxyConfig):
    """Create app and return (app, fake_pool) so tests can assert on the pool."""
    fake_pool = MagicMock(spec=ModelPool)
    fake_pool.list_loaded = MagicMock(return_value=[])
    fake_pool.get_or_load = AsyncMock()
    fake_pool.unload = AsyncMock()
    fake_pool.shutdown_all = AsyncMock()

    # Write a temporary key file so the auth middleware is satisfied.
    _td = tempfile.mkdtemp()
    key_file = str(pathlib.Path(_td) / "keys.txt")
    pathlib.Path(key_file).write_text(f"{_TEST_KEY}\n")

    with patch.dict(os.environ, {"VLLM_KEYS_FILE": key_file}):
        with patch("vllm_proxy.app.ModelPool", return_value=fake_pool):
            app = create_app(config)

    return app, fake_pool


# ---------------------------------------------------------------------------
# GET /admin/models
# ---------------------------------------------------------------------------


def test_list_admin_models_returns_all_configured():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        resp = client.get("/admin/models", headers=_AUTH_HEADER)
    assert resp.status_code == 200
    model_ids = [m["model_id"] for m in resp.json()["models"]]
    assert "model-a" in model_ids
    assert "model-b" in model_ids


def test_list_admin_models_shows_warm_flag():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    fake_pool.list_loaded.return_value = [{"model_id": "model-a", "port": 9000, "last_used": 0.0}]
    with TestClient(app) as client:
        resp = client.get("/admin/models", headers=_AUTH_HEADER)
    models = {m["model_id"]: m for m in resp.json()["models"]}
    assert models["model-a"]["warm"] is True
    assert models["model-b"]["warm"] is False


# ---------------------------------------------------------------------------
# POST /admin/models/{model_id} — upsert
# ---------------------------------------------------------------------------


def test_upsert_new_model():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        resp = client.post(
            "/admin/models/new-model",
            json={"model_path": "/models/new", "vllm_args": ["--dtype=float16"], "priority": 0},
            headers=_AUTH_HEADER,
        )
        assert resp.status_code == 200
        list_resp = client.get("/admin/models", headers=_AUTH_HEADER)
    model_ids = [m["model_id"] for m in list_resp.json()["models"]]
    assert "new-model" in model_ids


def test_upsert_replaces_existing_model():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        client.post(
            "/admin/models/model-a",
            json={"model_path": "/models/a-new", "vllm_args": [], "priority": 0},
            headers=_AUTH_HEADER,
        )
        list_resp = client.get("/admin/models", headers=_AUTH_HEADER)
    models = {m["model_id"]: m for m in list_resp.json()["models"]}
    assert models["model-a"]["model_path"] == "/models/a-new"


# ---------------------------------------------------------------------------
# DELETE /admin/models/{model_id}
# ---------------------------------------------------------------------------


def test_delete_model_removes_from_config():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        resp = client.delete("/admin/models/model-a", headers=_AUTH_HEADER)
        assert resp.status_code == 200
        list_resp = client.get("/admin/models", headers=_AUTH_HEADER)
    model_ids = [m["model_id"] for m in list_resp.json()["models"]]
    assert "model-a" not in model_ids
    assert "model-b" in model_ids


def test_delete_nonexistent_model_returns_404():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        resp = client.delete("/admin/models/does-not-exist", headers=_AUTH_HEADER)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /admin/models/{model_id}/load
# ---------------------------------------------------------------------------


def test_load_model_endpoint():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        resp = client.post("/admin/models/model-a/load", headers=_AUTH_HEADER)
    assert resp.status_code == 200
    fake_pool.get_or_load.assert_awaited_once_with("model-a")


def test_load_unknown_model_returns_404():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        resp = client.post("/admin/models/ghost/load", headers=_AUTH_HEADER)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /admin/models/{model_id}/unload
# ---------------------------------------------------------------------------


def test_unload_model_endpoint():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        resp = client.post("/admin/models/model-b/unload", headers=_AUTH_HEADER)
    assert resp.status_code == 200
    fake_pool.unload.assert_awaited_once_with("model-b")
