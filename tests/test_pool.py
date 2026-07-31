"""Tests for vllm_proxy.pool — LRU eviction, double-load guard, port recycling."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_proxy.config import ModelConfig, PoolConfig
from vllm_proxy.pool import ModelPool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_pool(max_size: int = 2, base_port: int = 9000) -> ModelPool:
    pool_cfg = PoolConfig(max_size=max_size, base_port=base_port, startup_timeout_seconds=10)
    models = {
        "model-a": ModelConfig(model_path="/models/a"),
        "model-b": ModelConfig(model_path="/models/b"),
        "model-c": ModelConfig(model_path="/models/c"),
    }
    return ModelPool(pool_config=pool_cfg, models=models, log_dir="/tmp/test-logs")


def _make_fake_process(model_id: str, port: int) -> MagicMock:
    proc = MagicMock()
    proc.model_id = model_id
    proc.port = port
    proc.stop = AsyncMock()
    return proc


# ---------------------------------------------------------------------------
# Helpers to inject a fake process without actually spawning VLLM
# ---------------------------------------------------------------------------


def _patch_vllm_start(fake_proc):
    """Return a context manager that patches VllmProcess.start."""
    return patch(
        "vllm_proxy.pool.VllmProcess.start",
        new_callable=AsyncMock,
        return_value=fake_proc,
    )


def _patch_wait_ready(result: bool = True):
    return patch.object(
        fake_proc := MagicMock(),
        "wait_until_ready",
        new_callable=AsyncMock,
        return_value=result,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_load_warms_model():
    pool = make_pool(max_size=2)
    fake = _make_fake_process("model-a", 9000)
    fake.wait_until_ready = AsyncMock(return_value=True)

    with _patch_vllm_start(fake):
        proc = await pool.get_or_load("model-a")

    assert proc is fake
    assert "model-a" in {e["model_id"] for e in pool.list_loaded()}


@pytest.mark.asyncio
async def test_second_call_returns_cached():
    pool = make_pool(max_size=2)
    fake = _make_fake_process("model-a", 9000)
    fake.wait_until_ready = AsyncMock(return_value=True)

    with _patch_vllm_start(fake):
        proc1 = await pool.get_or_load("model-a")
        # Second call — VllmProcess.start should NOT be called again.
        proc2 = await pool.get_or_load("model-a")

    assert proc1 is proc2


@pytest.mark.asyncio
async def test_lru_eviction_when_pool_full():
    """model-a loaded first, then model-b; adding model-c should evict model-a (LRU)."""
    pool = make_pool(max_size=2, base_port=9000)

    fake_a = _make_fake_process("model-a", 9000)
    fake_a.wait_until_ready = AsyncMock(return_value=True)

    fake_b = _make_fake_process("model-b", 9001)
    fake_b.wait_until_ready = AsyncMock(return_value=True)

    fake_c = _make_fake_process("model-c", 9000)  # port reused after eviction
    fake_c.wait_until_ready = AsyncMock(return_value=True)

    call_count = 0
    fakes = [fake_a, fake_b, fake_c]

    async def fake_start(model_id, model_path, vllm_args, port, log_dir, env=None):
        nonlocal call_count
        f = fakes[call_count]
        f.port = port  # use the actually allocated port
        call_count += 1
        return f

    with patch("vllm_proxy.pool.VllmProcess.start", side_effect=fake_start):
        await pool.get_or_load("model-a")
        await pool.get_or_load("model-b")
        # model-a is now LRU
        await pool.get_or_load("model-c")

    # model-a should have been evicted and stopped
    fake_a.stop.assert_awaited_once()
    loaded = {e["model_id"] for e in pool.list_loaded()}
    assert "model-a" not in loaded
    assert "model-b" in loaded
    assert "model-c" in loaded


@pytest.mark.asyncio
async def test_port_recycled_after_eviction():
    pool = make_pool(max_size=1, base_port=9000)
    ports_used: list[int] = []

    fake_a = _make_fake_process("model-a", 9000)
    fake_a.wait_until_ready = AsyncMock(return_value=True)
    fake_b = _make_fake_process("model-b", 9000)
    fake_b.wait_until_ready = AsyncMock(return_value=True)

    fakes = [fake_a, fake_b]
    call_count = 0

    async def fake_start(model_id, model_path, vllm_args, port, log_dir, env=None):
        nonlocal call_count
        f = fakes[call_count]
        f.port = port
        ports_used.append(port)
        call_count += 1
        return f

    with patch("vllm_proxy.pool.VllmProcess.start", side_effect=fake_start):
        await pool.get_or_load("model-a")
        await pool.get_or_load("model-b")  # evicts model-a, reuses port 9000

    assert ports_used == [9000, 9000]


@pytest.mark.asyncio
async def test_model_env_passed_to_vllm_process():
    pool_cfg = PoolConfig(max_size=1, base_port=9000, startup_timeout_seconds=10)
    models = {"model-a": ModelConfig(model_path="/models/a", env={"VLLM_USE_DEEP_GEMM": "0"})}
    pool = ModelPool(pool_config=pool_cfg, models=models, log_dir="/tmp/test-logs")

    fake = _make_fake_process("model-a", 9000)
    fake.wait_until_ready = AsyncMock(return_value=True)

    with _patch_vllm_start(fake) as mock_start:
        await pool.get_or_load("model-a")

    assert mock_start.call_args.kwargs["env"] == {"VLLM_USE_DEEP_GEMM": "0"}


@pytest.mark.asyncio
async def test_unknown_model_raises_key_error():
    pool = make_pool()
    with pytest.raises(KeyError):
        await pool.get_or_load("does-not-exist")


@pytest.mark.asyncio
async def test_explicit_unload():
    pool = make_pool(max_size=2)
    fake = _make_fake_process("model-a", 9000)
    fake.wait_until_ready = AsyncMock(return_value=True)

    with _patch_vllm_start(fake):
        await pool.get_or_load("model-a")

    await pool.unload("model-a")

    fake.stop.assert_awaited_once()
    loaded = {e["model_id"] for e in pool.list_loaded()}
    assert "model-a" not in loaded


@pytest.mark.asyncio
async def test_shutdown_all():
    pool = make_pool(max_size=2)

    fake_a = _make_fake_process("model-a", 9000)
    fake_a.wait_until_ready = AsyncMock(return_value=True)
    fake_b = _make_fake_process("model-b", 9001)
    fake_b.wait_until_ready = AsyncMock(return_value=True)

    fakes = [fake_a, fake_b]
    call_count = 0

    async def fake_start(model_id, model_path, vllm_args, port, log_dir, env=None):
        nonlocal call_count
        f = fakes[call_count]
        f.port = port
        call_count += 1
        return f

    with patch("vllm_proxy.pool.VllmProcess.start", side_effect=fake_start):
        await pool.get_or_load("model-a")
        await pool.get_or_load("model-b")

    await pool.shutdown_all()

    fake_a.stop.assert_awaited_once()
    fake_b.stop.assert_awaited_once()
    assert pool.list_loaded() == []
