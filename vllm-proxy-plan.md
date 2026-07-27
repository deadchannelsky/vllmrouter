# vllm-proxy — Plan

## Top-Level Overview

Build a Python/FastAPI proxy service called **vllm-proxy** that sits in front of one or more VLLM subprocess instances. The proxy presents a fully OpenAI-compatible API to callers, automatically managing model loading and unloading behind the scenes. A configurable warm pool keeps up to N models loaded simultaneously on separate ports; when the pool is full and a new model is needed, the least-recently-used model is evicted. Per-model VLLM CLI parameters are defined in a YAML config file and can be overridden at runtime via an admin REST API (changes are ephemeral — restart reloads from YAML).

**Model identity design:** Each model is given a short friendly alias (e.g. `mistral-7b`) in config that maps to a full disk path (e.g. `/models/mistral-7b`). Callers use the alias in the `"model"` field of their requests. `GET /v1/models` returns aliases. Before forwarding a request to VLLM, the proxy rewrites the `"model"` field in the request body from the alias to the full disk path, since VLLM identifies models by their path. VLLM is started via `vllm serve <model_path> --port <port> [vllm_args...]`.

**Non-goals:**
- Authentication / API key enforcement
- Persistent admin API state
- Multi-GPU sharding coordination (left to VLLM's own flags)
- Health / readiness probes

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffold

**Intent**  
Create the Python package layout, dependency manifest, and CLI entry point so the project is runnable from day one and all subsequent sub-tasks have a stable place to land code.

**Expected Outcomes**  
- `pyproject.toml` and `requirements.txt` present with all required dependencies declared
- `vllm_proxy/` package directory with `__init__.py`
- `vllm_proxy/__main__.py` CLI entry point that starts the FastAPI app via Uvicorn with `--host`, `--port`, and `--config` flags
- Running `python -m vllm_proxy --config config.yaml` starts the server

**Todo List**  
1. Create the top-level directory structure: `vllm_proxy/`, `logs/`, `tests/`
2. Write `pyproject.toml` declaring the package, entry point (`vllm-proxy`), and metadata
3. Write `requirements.txt` with pinned/minimum versions: `fastapi`, `uvicorn[standard]`, `httpx`, `pyyaml`, `anyio`
4. Write `vllm_proxy/__init__.py` (empty or version string)
5. Write `vllm_proxy/__main__.py` with argparse CLI (`--host`, `--port`, `--config`) that imports and launches the app

**Relevant Context**  
- No existing codebase — greenfield project
- Entry point name: `vllm-proxy`
- Default config path: `config.yaml`

**Status:** `[ ] pending`

---

### Sub-Task 2 — Configuration Schema & Loader

**Intent**  
Define the YAML config schema and a Python dataclass/model layer that loads, validates, and exposes config at startup. This is the single source of truth for pool settings and per-model VLLM parameters.

**Expected Outcomes**  
- `vllm_proxy/config.py` with dataclasses (or Pydantic models) for `ProxyConfig`, `PoolConfig`, and `ModelConfig`
- `load_config(path: str) -> ProxyConfig` function that reads YAML and returns a validated config object
- A documented `config.yaml` example file at the repo root

**Todo List**  
1. Define `PoolConfig` dataclass: `max_size: int`, `base_port: int`, `startup_timeout_seconds: int`
2. Define `ModelConfig` dataclass: `model_id: str`, `model_path: str`, `vllm_args: list[str]`, `priority: int` (reserved, default 0)
3. Define `ProxyConfig` dataclass: `host: str`, `port: int`, `log_dir: str`, `pool: PoolConfig`, `models: dict[str, ModelConfig]`
4. Implement `load_config` with clear error messages for missing required fields
5. Write `config.yaml` example covering at least two models with differing `vllm_args`

**Example config.yaml shape:**
```yaml
host: "0.0.0.0"
port: 8000
log_dir: "logs"

pool:
  max_size: 2
  base_port: 9000
  startup_timeout_seconds: 120

models:
  mistral-7b:
    model_path: "/models/mistral-7b"
    vllm_args:
      - "--dtype=float16"
      - "--max-model-len=8192"
  llama3-8b:
    model_path: "/models/llama3-8b"
    vllm_args:
      - "--dtype=bfloat16"
      - "--tensor-parallel-size=2"
```

**Relevant Context**  
- `vllm_proxy/config.py`
- `config.yaml` (example at repo root)

**Status:** `[ ] pending`

---

### Sub-Task 3 — VLLM Process Manager

**Intent**  
Implement the component responsible for spawning, monitoring, and terminating individual VLLM subprocess instances. Each instance is assigned a port from the pool's port range. Logs go to `<log_dir>/<model_id>.log`.

**Expected Outcomes**  
- `vllm_proxy/process_manager.py` with class `VllmProcess`
- `VllmProcess` encapsulates: `model_id`, `port`, `process` (asyncio subprocess), `log_file_handle`
- `start(model_config, port, log_dir) -> VllmProcess` async factory — builds the `vllm serve` command, opens the log file, spawns the subprocess
- `stop() -> None` async method — sends SIGTERM, waits for exit with timeout, then SIGKILL
- `wait_until_ready(timeout) -> bool` async method — polls the VLLM process's `/health` endpoint until it returns 200 or timeout expires

**Todo List**  
1. Implement command builder: `["vllm", "serve", model_path, "--port", str(port)] + vllm_args`
2. Open `<log_dir>/<model_id>.log` in append mode and pass as stdout/stderr to subprocess
3. Implement `wait_until_ready` using `httpx.AsyncClient` polling loop with configurable interval (1 second)
4. Implement `stop` with graceful SIGTERM → wait → SIGKILL fallback
5. Log process start/stop events to proxy stdout (Python `logging`)

**Relevant Context**  
- `vllm_proxy/process_manager.py`
- VLLM CLI entry point: `vllm serve <model_path> --port <port> [vllm_args...]`
- VLLM exposes `GET /health` for readiness

**Status:** `[ ] pending`

---

### Sub-Task 4 — Warm Pool & LRU Router

**Intent**  
Implement the pool that manages up to N simultaneously running VLLM processes, tracks LRU order, and handles eviction + cold-start when a new model is requested.

**Expected Outcomes**  
- `vllm_proxy/pool.py` with class `ModelPool`
- `get_or_load(model_id) -> VllmProcess` async method — returns a running process for the model, loading it if needed, evicting LRU if pool is full
- `unload(model_id) -> None` async method — explicitly evicts a model
- `list_loaded() -> list[dict]` — returns current pool state (model_id, port, last_used)
- Port allocation: maintain a set of available ports derived from `base_port` to `base_port + max_size - 1`
- Thread-safe: use `asyncio.Lock` to prevent concurrent load races for the same model
- LRU tracking via `collections.OrderedDict` or equivalent

**Todo List**  
1. Initialize pool from `PoolConfig`; pre-compute available port set
2. Implement `get_or_load`: check if model already warm → return immediately (update LRU timestamp); else acquire lock, check again (double-check), evict LRU if full, start new process, await `wait_until_ready`, register in pool
3. Implement eviction: call `VllmProcess.stop()`, release port back to available set, remove from pool dict
4. Implement `unload` for explicit eviction
5. Expose `list_loaded` for admin API use

**Relevant Context**  
- `vllm_proxy/pool.py`
- Depends on `VllmProcess` from Sub-Task 3 and `PoolConfig` from Sub-Task 2
- Lock must be per-model to allow concurrent requests for different models to proceed in parallel

**Status:** `[ ] pending`

---

### Sub-Task 5 — OpenAI-Compatible Proxy Endpoints

**Intent**  
Implement the FastAPI application with a generic `/v1/*` catch-all that extracts the model ID from the request body, ensures the model is warm via the pool, then reverse-proxies the full request (including SSE streaming) to the correct VLLM port.

**Expected Outcomes**  
- `vllm_proxy/app.py` with a FastAPI app instance
- Catch-all route `ANY /v1/{path:path}` that handles all VLLM-compatible endpoints
- Model ID extraction from JSON body field `"model"` (present in chat/completions, completions, embeddings, tokenize); fall back to first loaded model or 422 if body has no `model` field and pool is empty
- Full request forwarding: method, headers, body, query params
- SSE streaming passthrough using `StreamingResponse` when the upstream response is `text/event-stream`
- Non-streaming responses returned as-is with original status code

**Todo List**
1. Create `vllm_proxy/app.py`; define `create_app(config, pool) -> FastAPI`
2. Implement model extraction helper: parse JSON body, read `model` key (alias)
3. Implement the catch-all route; call `pool.get_or_load(model_id)` to get the target port
4. Before forwarding, rewrite the `"model"` field in the request body from the alias (e.g. `mistral-7b`) to the full disk path (e.g. `/models/mistral-7b`) so VLLM accepts the request
5. Forward the rewritten request via `httpx.AsyncClient` to `http://127.0.0.1:{port}/v1/{path}`
6. Detect `Content-Type: text/event-stream` in upstream response and return `StreamingResponse` with async byte iterator
7. For non-streaming, return `Response` with upstream status, headers, and body
8. Handle `model not found in config` with a clear 404 JSON error
9. Implement `GET /v1/models` as a special case — synthesize the response from the proxy's config (return aliases), do not pass through to VLLM

**Relevant Context**  
- `vllm_proxy/app.py`
- Depends on `ModelPool` from Sub-Task 4
- `httpx.AsyncClient` supports streaming via `stream()` context manager
- FastAPI `StreamingResponse` accepts an async generator

**Status:** `[ ] pending`

---

### Sub-Task 6 — Admin REST API

**Intent**  
Add an `/admin/*` router to the FastAPI app that exposes pool introspection and runtime model config management. Changes to model configs are held in memory only (ephemeral).

**Expected Outcomes**  
- `vllm_proxy/admin.py` with an `APIRouter` mounted at `/admin`
- `GET /admin/models` — list all known model configs + which are currently warm
- `POST /admin/models/{model_id}` — add or replace a model config at runtime (body is a `ModelConfig` JSON object)
- `DELETE /admin/models/{model_id}` — remove a model config; if currently warm, unload first
- `POST /admin/models/{model_id}/load` — explicitly pre-warm a model
- `POST /admin/models/{model_id}/unload` — explicitly evict a model from the pool

**Todo List**  
1. Create `vllm_proxy/admin.py` with `create_admin_router(config, pool) -> APIRouter`
2. Implement `GET /admin/models`: merge static config models with pool's live state
3. Implement `POST /admin/models/{model_id}`: validate body as `ModelConfig`, upsert into the in-memory config dict
4. Implement `DELETE /admin/models/{model_id}`: unload if warm, remove from config dict
5. Implement `POST /admin/models/{model_id}/load`: call `pool.get_or_load`
6. Implement `POST /admin/models/{model_id}/unload`: call `pool.unload`
7. Mount the admin router in `app.py`

**Relevant Context**  
- `vllm_proxy/admin.py`
- Runtime model config changes must update the same in-memory dict that `ModelPool` reads from — pass a shared mutable `dict` reference

**Status:** `[ ] pending`

---

### Sub-Task 7 — Wiring & Startup Lifecycle

**Intent**  
Wire all components together in `__main__.py` and `app.py`, ensuring proper async startup/shutdown hooks (pool cleanup on exit) and the config is threaded through to every component correctly.

**Expected Outcomes**  
- On startup: config loaded, `ModelPool` initialized, `httpx.AsyncClient` created, app and admin router mounted
- On shutdown: all warm VLLM processes gracefully stopped
- `python -m vllm_proxy --config config.yaml` starts cleanly and logs the listening address and loaded config summary

**Todo List**  
1. Add FastAPI `lifespan` context manager to `app.py`: create pool and httpx client on enter, call `pool.shutdown_all()` on exit
2. Add `shutdown_all()` to `ModelPool`: iterate and stop all warm processes
3. Pass `httpx.AsyncClient` instance into the catch-all route (via app state or dependency injection)
4. Update `__main__.py` to load config, pass it to `create_app`, then hand off to `uvicorn.run`
5. Log startup summary: proxy address, pool max size, number of configured models

**Relevant Context**  
- `vllm_proxy/__main__.py`
- `vllm_proxy/app.py`
- `vllm_proxy/pool.py`
- FastAPI lifespan: `@asynccontextmanager` pattern

**Status:** `[ ] pending`

---

### Sub-Task 8 — Tests

**Intent**  
Add a lightweight test suite covering the critical path: config loading, LRU eviction logic, model ID extraction, and admin API endpoints. VLLM process spawning is mocked.

**Expected Outcomes**  
- `tests/` directory with at least: `test_config.py`, `test_pool.py`, `test_proxy.py`, `test_admin.py`
- All tests pass with `pytest`
- No real VLLM processes spawned during tests

**Todo List**  
1. `test_config.py` — test `load_config` with valid YAML, missing required fields, and unknown fields
2. `test_pool.py` — test LRU eviction (mock `VllmProcess`), double-load guard, port recycling
3. `test_proxy.py` — test model extraction helper, 422 on missing model, 404 on unknown model; mock httpx for forwarding
4. `test_admin.py` — test each admin endpoint using FastAPI `TestClient`; verify ephemeral config mutations

**Relevant Context**  
- `tests/`
- Use `pytest`, `pytest-asyncio`, and `unittest.mock`
- Add test dependencies to `requirements.txt` under a `[test]` comment or separate `requirements-dev.txt`

**Status:** `[ ] pending`

---

## File Layout

```
vllm-proxy/
├── pyproject.toml
├── requirements.txt
├── config.yaml              # example config
├── logs/                    # VLLM per-model log files (gitignored)
├── vllm_proxy/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── config.py            # config schema + loader
│   ├── process_manager.py   # VllmProcess
│   ├── pool.py              # ModelPool + LRU
│   ├── app.py               # FastAPI app + catch-all proxy route
│   └── admin.py             # Admin APIRouter
└── tests/
    ├── test_config.py
    ├── test_pool.py
    ├── test_proxy.py
    └── test_admin.py
```

---

## Dependency Summary

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn[standard]` | ASGI server |
| `httpx` | Async HTTP client for proxying to VLLM |
| `pyyaml` | YAML config loading |
| `anyio` | Async subprocess support |
| `pytest` | Test runner |
| `pytest-asyncio` | Async test support |
| `httpx` | TestClient support for FastAPI |
