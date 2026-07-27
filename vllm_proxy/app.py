"""FastAPI application for vllm-proxy.

Exposes an OpenAI-compatible ``/v1/*`` catch-all that:
1. Extracts the ``model`` alias from the JSON request body.
2. Ensures the model is warm via :class:`~vllm_proxy.pool.ModelPool`.
3. Rewrites ``model`` in the body from alias → disk path.
4. Reverse-proxies the full request (including SSE streaming) to the
   correct VLLM port.

``GET /v1/models`` is handled locally — it returns the proxy's configured
model aliases rather than passing through to VLLM.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_proxy.config import ProxyConfig
from vllm_proxy.pool import ModelPool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: ProxyConfig) -> FastAPI:
    """Create and return the FastAPI application."""

    pool = ModelPool(
        pool_config=config.pool,
        models=config.models,
        log_dir=config.log_dir,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup
        client = httpx.AsyncClient(timeout=None)
        app.state.pool = pool
        app.state.config = config
        app.state.http_client = client
        logger.info(
            "vllm-proxy ready  |  pool_max=%d  |  models=%s",
            config.pool.max_size,
            list(config.models.keys()),
        )
        yield
        # Shutdown
        logger.info("Shutting down — stopping all VLLM processes…")
        await pool.shutdown_all()
        await client.aclose()

    app = FastAPI(title="vllm-proxy", lifespan=lifespan)

    from vllm_proxy.admin import create_admin_router

    admin_router = create_admin_router(config, pool)
    app.include_router(admin_router)

    # ------------------------------------------------------------------
    # GET /v1/models — synthesized from proxy config (not forwarded)
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        now = int(time.time())
        data = [
            {
                "id": model_id,
                "object": "model",
                "created": now,
                "owned_by": "vllm-proxy",
            }
            for model_id in request.app.state.config.models
        ]
        return JSONResponse({"object": "list", "data": data})

    # ------------------------------------------------------------------
    # Catch-all for all other /v1/* endpoints
    # ------------------------------------------------------------------

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def proxy(path: str, request: Request) -> Response:
        cfg: ProxyConfig = request.app.state.config
        _pool: ModelPool = request.app.state.pool
        client: httpx.AsyncClient = request.app.state.http_client

        # --- Parse body ------------------------------------------------
        raw_body = await request.body()
        body_dict: dict | None = None
        model_id: str | None = None

        if raw_body:
            try:
                body_dict = json.loads(raw_body)
                model_id = body_dict.get("model") if isinstance(body_dict, dict) else None
            except json.JSONDecodeError:
                pass  # non-JSON body (e.g. raw completions) — no model rewrite

        # Fall back: if no model in body but something is warm, use that.
        if not model_id:
            loaded = _pool.list_loaded()
            if loaded:
                model_id = loaded[0]["model_id"]
            else:
                return JSONResponse(
                    status_code=422,
                    content={"detail": "No 'model' field in request body and no model is currently warm."},
                )

        if model_id not in cfg.models:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Model '{model_id}' not found in proxy config."},
            )

        # --- Ensure model is warm --------------------------------------
        try:
            proc = await _pool.get_or_load(model_id)
        except RuntimeError as exc:
            return JSONResponse(status_code=503, content={"detail": str(exc)})

        # --- Rewrite model alias → disk path in body -------------------
        if body_dict is not None and isinstance(body_dict, dict) and "model" in body_dict:
            body_dict["model"] = cfg.models[model_id].model_path
            raw_body = json.dumps(body_dict).encode()

        # --- Forward request ------------------------------------------
        target_url = f"http://127.0.0.1:{proc.port}/v1/{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        # Strip hop-by-hop headers; keep everything else.
        forward_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "transfer-encoding", "connection"}
        }
        if raw_body:
            forward_headers["content-length"] = str(len(raw_body))

        upstream_request = client.build_request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            content=raw_body,
        )

        upstream_response = await client.send(upstream_request, stream=True)

        content_type = upstream_response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            # SSE streaming passthrough
            async def stream_bytes() -> AsyncIterator[bytes]:
                async for chunk in upstream_response.aiter_bytes():
                    yield chunk
                await upstream_response.aclose()

            return StreamingResponse(
                stream_bytes(),
                status_code=upstream_response.status_code,
                headers=dict(upstream_response.headers),
                media_type="text/event-stream",
            )

        # Non-streaming — read full body then close.
        content = await upstream_response.aread()
        await upstream_response.aclose()
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=dict(upstream_response.headers),
            media_type=content_type or None,
        )

    return app
