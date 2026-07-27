"""Admin REST API router for vllm-proxy.

Mounted at ``/admin``; provides pool introspection and runtime model config
management.  All changes are ephemeral — restart reloads from YAML.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm_proxy.config import ModelConfig, ProxyConfig
from vllm_proxy.pool import ModelPool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request body schema (Pydantic for validation)
# ---------------------------------------------------------------------------


class ModelConfigBody(BaseModel):
    model_path: str
    vllm_args: list[str] = []
    priority: int = 0


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_admin_router(config: ProxyConfig, pool: ModelPool) -> APIRouter:
    """Return an :class:`APIRouter` with all admin endpoints."""

    router = APIRouter(prefix="/admin", tags=["admin"])

    # Shared mutable reference so both admin endpoints and ModelPool see updates.
    models = config.models  # dict[str, ModelConfig]

    # ------------------------------------------------------------------
    # GET /admin/models
    # ------------------------------------------------------------------

    @router.get("/models")
    async def list_models() -> JSONResponse:
        """List all configured models with their warm state."""
        loaded_ids = {entry["model_id"] for entry in pool.list_loaded()}
        result = []
        for model_id, cfg in models.items():
            result.append(
                {
                    "model_id": model_id,
                    "model_path": cfg.model_path,
                    "vllm_args": cfg.vllm_args,
                    "priority": cfg.priority,
                    "warm": model_id in loaded_ids,
                }
            )
        return JSONResponse({"models": result})

    # ------------------------------------------------------------------
    # POST /admin/models/{model_id}
    # ------------------------------------------------------------------

    @router.post("/models/{model_id}")
    async def upsert_model(model_id: str, body: ModelConfigBody) -> JSONResponse:
        """Add or replace a model config at runtime (ephemeral)."""
        models[model_id] = ModelConfig(
            model_path=body.model_path,
            vllm_args=body.vllm_args,
            priority=body.priority,
        )
        logger.info("Admin: upserted model config for '%s'", model_id)
        return JSONResponse({"status": "ok", "model_id": model_id})

    # ------------------------------------------------------------------
    # DELETE /admin/models/{model_id}
    # ------------------------------------------------------------------

    @router.delete("/models/{model_id}")
    async def delete_model(model_id: str) -> JSONResponse:
        """Remove a model config; unloads from pool if currently warm."""
        if model_id not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
        await pool.unload(model_id)
        del models[model_id]
        logger.info("Admin: deleted model config for '%s'", model_id)
        return JSONResponse({"status": "ok", "model_id": model_id})

    # ------------------------------------------------------------------
    # POST /admin/models/{model_id}/load
    # ------------------------------------------------------------------

    @router.post("/models/{model_id}/load")
    async def load_model(model_id: str) -> JSONResponse:
        """Explicitly pre-warm a model."""
        if model_id not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
        try:
            await pool.get_or_load(model_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        logger.info("Admin: pre-warmed model '%s'", model_id)
        return JSONResponse({"status": "ok", "model_id": model_id})

    # ------------------------------------------------------------------
    # POST /admin/models/{model_id}/unload
    # ------------------------------------------------------------------

    @router.post("/models/{model_id}/unload")
    async def unload_model(model_id: str) -> JSONResponse:
        """Explicitly evict a model from the pool."""
        if model_id not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
        await pool.unload(model_id)
        logger.info("Admin: unloaded model '%s'", model_id)
        return JSONResponse({"status": "ok", "model_id": model_id})

    return router
