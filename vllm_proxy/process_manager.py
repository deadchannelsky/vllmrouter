"""VLLM process manager for vllm-proxy.

Provides :class:`VllmProcess`, which wraps a single ``vllm serve`` subprocess.
Use the :func:`VllmProcess.start` async factory to spawn a process; call
:meth:`VllmProcess.stop` to terminate it gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from typing import IO

import httpx

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 1.0  # seconds between /health polls


# ---------------------------------------------------------------------------
# VllmProcess
# ---------------------------------------------------------------------------


@dataclass
class VllmProcess:
    model_id: str
    port: int
    process: asyncio.subprocess.Process
    _log_handle: IO[bytes] = field(repr=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def start(
        cls,
        model_id: str,
        model_path: str,
        vllm_args: list[str],
        port: int,
        log_dir: str,
        env: dict[str, str] | None = None,
    ) -> "VllmProcess":
        """Spawn ``vllm serve`` and return a :class:`VllmProcess` instance."""
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{model_id}.log")
        log_handle = open(log_path, "ab")  # append mode, binary

        cmd = ["vllm", "serve", model_path, "--port", str(port)] + vllm_args

        logger.info("Starting VLLM process for '%s' on port %d: %s", model_id, port, " ".join(cmd))

        subprocess_env = {**os.environ, **(env or {})}

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_handle,
            stderr=log_handle,
            env=subprocess_env,
        )

        logger.info("VLLM process for '%s' spawned with PID %d", model_id, process.pid)
        return cls(model_id=model_id, port=port, process=process, _log_handle=log_handle)

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    async def wait_until_ready(self, timeout: float) -> bool:
        """Poll ``/health`` until a 200 is received or *timeout* seconds elapse.

        Returns ``True`` if the process became ready, ``False`` on timeout.
        """
        url = f"http://127.0.0.1:{self.port}/health"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        async with httpx.AsyncClient() as client:
            while loop.time() < deadline:
                # Also bail early if the process died.
                if self.process.returncode is not None:
                    logger.error(
                        "VLLM process for '%s' exited with code %d before becoming ready",
                        self.model_id,
                        self.process.returncode,
                    )
                    return False
                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        logger.info("VLLM process for '%s' is ready on port %d", self.model_id, self.port)
                        return True
                except (httpx.RequestError, httpx.HTTPStatusError):
                    pass
                await asyncio.sleep(_HEALTH_POLL_INTERVAL)

        logger.warning(
            "VLLM process for '%s' did not become ready within %.0f seconds",
            self.model_id,
            timeout,
        )
        return False

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def stop(self, graceful_timeout: float = 10.0) -> None:
        """Send SIGTERM; if the process does not exit within *graceful_timeout*,
        send SIGKILL.
        """
        if self.process.returncode is not None:
            logger.debug("VLLM process for '%s' already exited", self.model_id)
            self._log_handle.close()
            return

        logger.info("Stopping VLLM process for '%s' (PID %d)", self.model_id, self.process.pid)
        try:
            self.process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone

        try:
            await asyncio.wait_for(self.process.wait(), timeout=graceful_timeout)
            logger.info("VLLM process for '%s' exited gracefully", self.model_id)
        except asyncio.TimeoutError:
            logger.warning(
                "VLLM process for '%s' did not exit within %.0f seconds; sending SIGKILL",
                self.model_id,
                graceful_timeout,
            )
            try:
                self.process.send_signal(signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.process.wait()
            logger.info("VLLM process for '%s' killed", self.model_id)
        finally:
            self._log_handle.close()
