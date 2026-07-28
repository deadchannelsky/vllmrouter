"""Tests for vllm_proxy.app — proxy routing, model extraction, error cases."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllm_proxy.config import ModelConfig, PoolConfig, ProxyConfig
from vllm_proxy.app import create_app
from vllm_proxy.pool import ModelPool

# A stable test key injected into every test via the patched key file.
_TEST_KEY = "sk-test-proxy"
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
            "mistral-7b": ModelConfig(model_path="/models/mistral-7b", vllm_args=["--dtype=float16"]),
            "llama3-8b": ModelConfig(model_path="/models/llama3-8b"),
        },
    )


def make_fake_process(port: int = 9000) -> MagicMock:
    proc = MagicMock()
    proc.port = port
    return proc


def _make_app_with_fake_pool(config: ProxyConfig, tmp_path=None):
    """Return (app, fake_pool) with ModelPool patched at creation time."""
    import tempfile, pathlib

    fake_pool = MagicMock(spec=ModelPool)
    fake_pool.list_loaded = MagicMock(return_value=[])
    fake_pool.get_or_load = AsyncMock()
    fake_pool.unload = AsyncMock()
    fake_pool.shutdown_all = AsyncMock()

    # Write a temporary key file so the auth middleware is satisfied.
    if tmp_path is None:
        _td = tempfile.mkdtemp()
        key_file = str(pathlib.Path(_td) / "keys.txt")
    else:
        key_file = str(tmp_path / "keys.txt")
    pathlib.Path(key_file).write_text(f"{_TEST_KEY}\n")

    with patch.dict(os.environ, {"VLLM_KEYS_FILE": key_file}):
        with patch("vllm_proxy.app.ModelPool", return_value=fake_pool):
            app = create_app(config)

    return app, fake_pool


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


def test_list_models_returns_aliases():
    config = make_config()
    app, _ = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        resp = client.get("/v1/models", headers=_AUTH_HEADER)
    assert resp.status_code == 200
    data = resp.json()
    ids = [m["id"] for m in data["data"]]
    assert "mistral-7b" in ids
    assert "llama3-8b" in ids


# ---------------------------------------------------------------------------
# 404 on unknown model
# ---------------------------------------------------------------------------


def test_unknown_model_returns_404():
    config = make_config()
    app, _ = _make_app_with_fake_pool(config)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "no-such-model", "messages": []},
            headers=_AUTH_HEADER,
        )
    assert resp.status_code == 404
    assert "no-such-model" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 422 when body has no model and pool is empty
# ---------------------------------------------------------------------------


def test_no_model_and_empty_pool_returns_422():
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)
    fake_pool.list_loaded.return_value = []
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers=_AUTH_HEADER,
        )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Successful forwarding — non-streaming, verifies model alias rewrite
# ---------------------------------------------------------------------------


def test_proxy_forwards_request_and_rewrites_model():
    """The model alias in the body should be rewritten to the disk path."""
    config = make_config()
    app, fake_pool = _make_app_with_fake_pool(config)

    fake_proc = make_fake_process(port=9000)
    fake_pool.get_or_load = AsyncMock(return_value=fake_proc)
    fake_pool.list_loaded.return_value = [{"model_id": "mistral-7b"}]

    upstream_body = json.dumps({"id": "chatcmpl-test", "choices": []}).encode()
    sent_bodies: list[bytes] = []

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aread(self):
            return upstream_body

        async def aclose(self):
            pass

    async def fake_send(self, request, *, stream=False):
        sent_bodies.append(await request.aread())
        return FakeResp()

    import httpx as _httpx

    with patch.object(_httpx.AsyncClient, "send", new=fake_send):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "mistral-7b", "messages": [{"role": "user", "content": "hi"}]},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    # Verify the body was rewritten to the disk path
    assert len(sent_bodies) == 1
    sent = json.loads(sent_bodies[0])
    assert sent["model"] == "/models/mistral-7b"
