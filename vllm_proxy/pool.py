"""Warm pool with LRU eviction for vllm-proxy.

Maintains up to ``pool_config.max_size`` simultaneously running
:class:`~vllm_proxy.process_manager.VllmProcess` instances.  When the pool is
full and a new model is requested, the least-recently-used model is evicted.

Per-model locks prevent concurrent cold-start races for the same model_id.
Requests for *different* models that both arrive during a cold start proceed
independently (they hold different per-model locks).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from vllm_proxy.config import ModelConfig, PoolConfig
from vllm_proxy.process_manager import VllmProcess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal entry stored in the pool
# ---------------------------------------------------------------------------


@dataclass
class _PoolEntry:
    process: VllmProcess
    last_used: float  # unix timestamp


# ---------------------------------------------------------------------------
# ModelPool
# ---------------------------------------------------------------------------


class ModelPool:
    """Manages a warm pool of VLLM subprocess instances."""

    def __init__(
        self,
        pool_config: PoolConfig,
        models: dict[str, ModelConfig],
        log_dir: str,
    ) -> None:
        self._pool_config = pool_config
        # Mutable reference — admin API can add/remove models at runtime.
        self._models = models
        self._log_dir = log_dir

        # Available ports (set); the pool owns these.
        self._available_ports: set[int] = {
            pool_config.base_port + i for i in range(pool_config.max_size)
        }

        # Ordered by insertion time; moved to end on each access (LRU head = oldest).
        self._pool: OrderedDict[str, _PoolEntry] = OrderedDict()

        # Per-model locks — created lazily.
        self._model_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()  # protects _model_locks dict itself

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_load(self, model_id: str) -> VllmProcess:
        """Return a warm :class:`VllmProcess` for *model_id*.

        Loads the model if it is not already warm.  Evicts the LRU model if
        the pool is at capacity.

        Raises :class:`KeyError` if *model_id* is not in the config.
        """
        if model_id not in self._models:
            raise KeyError(model_id)

        # Always acquire the per-model lock — protects both the warm fast-path
        # (_touch) and the cold-start path against concurrent unload/eviction.
        lock = await self._get_model_lock(model_id)
        async with lock:
            # Already warm — just update LRU and return.
            if model_id in self._pool:
                self._touch(model_id)
                return self._pool[model_id].process

            # Evict LRU if pool is full.
            if len(self._pool) >= self._pool_config.max_size:
                await self._evict_lru()

            port = self._available_ports.pop()
            port_returned = False  # track whether the except handler needs to return it
            model_cfg = self._models[model_id]

            try:
                proc = await VllmProcess.start(
                    model_id=model_id,
                    model_path=model_cfg.model_path,
                    vllm_args=model_cfg.vllm_args,
                    port=port,
                    log_dir=self._log_dir,
                )
                ready = await proc.wait_until_ready(
                    timeout=self._pool_config.startup_timeout_seconds
                )
                if not ready:
                    await proc.stop()
                    self._available_ports.add(port)
                    port_returned = True
                    raise RuntimeError(
                        f"Model '{model_id}' failed to become ready within "
                        f"{self._pool_config.startup_timeout_seconds}s"
                    )
            except Exception:
                # Return the port only if it hasn't already been returned above.
                if not port_returned:
                    self._available_ports.add(port)
                raise

            self._pool[model_id] = _PoolEntry(process=proc, last_used=time.monotonic())
            self._pool.move_to_end(model_id)  # mark as most recently used
            logger.info("Model '%s' added to warm pool (pool size: %d)", model_id, len(self._pool))
            return proc

    async def unload(self, model_id: str) -> None:
        """Explicitly evict *model_id* from the pool.

        No-op if the model is not currently warm.
        """
        if model_id not in self._pool:
            return
        lock = await self._get_model_lock(model_id)
        async with lock:
            if model_id not in self._pool:
                return
            await self._evict(model_id)

    def list_loaded(self) -> list[dict[str, Any]]:
        """Return a snapshot of the current pool state."""
        return [
            {
                "model_id": model_id,
                "port": entry.process.port,
                "last_used": entry.last_used,
            }
            for model_id, entry in self._pool.items()
        ]

    async def shutdown_all(self) -> None:
        """Stop all warm processes — called during proxy shutdown."""
        model_ids = list(self._pool.keys())
        for model_id in model_ids:
            await self._evict(model_id)
        logger.info("All VLLM processes stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _touch(self, model_id: str) -> None:
        """Update LRU order and last_used timestamp for an already-warm model."""
        self._pool[model_id].last_used = time.monotonic()
        self._pool.move_to_end(model_id)

    async def _evict_lru(self) -> None:
        """Evict the least-recently-used (first) entry from the pool.

        Acquires the victim's per-model lock so that any coroutine already past
        the warm fast-path for that model finishes (or at least sees a clean
        stop) before the process is terminated.
        """
        lru_model_id, _ = next(iter(self._pool.items()))
        logger.info("Evicting LRU model '%s' from pool", lru_model_id)
        victim_lock = await self._get_model_lock(lru_model_id)
        async with victim_lock:
            await self._evict(lru_model_id)

    async def _evict(self, model_id: str) -> None:
        """Remove *model_id* from the pool and stop its process.

        Must be called while the caller holds the per-model lock for *model_id*
        (or during shutdown where no concurrent access is possible).
        """
        entry = self._pool.pop(model_id, None)
        if entry is None:
            return
        port = entry.process.port
        await entry.process.stop()
        self._available_ports.add(port)
        logger.info(
            "Model '%s' evicted; port %d returned to pool (pool size: %d)",
            model_id,
            port,
            len(self._pool),
        )

    async def _get_model_lock(self, model_id: str) -> asyncio.Lock:
        async with self._global_lock:
            if model_id not in self._model_locks:
                self._model_locks[model_id] = asyncio.Lock()
            return self._model_locks[model_id]
