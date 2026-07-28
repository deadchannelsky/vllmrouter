# API Key Authorization — Plan

## Top-Level Overview

Retrofit API key authorization into the vllm-proxy FastAPI application.
Accepted keys are managed externally and written to a file whose path is set
in the `VLLM_KEYS_FILE` environment variable.  Every incoming request to any
route (`/v1/*` and `/admin/*`) must carry a valid `Authorization: Bearer <key>`
header that matches a key present in that file.  The file is re-read on every
request so that key additions and removals made by the external manager take
effect immediately without restarting the proxy.

**Non-goals:**
- No in-memory key cache.
- No opt-out / bypass mode.
- No separate secret for admin routes.

---

## Sub-Tasks

---

### Sub-Task 1 — Create `vllm_proxy/auth.py`

**Status:** [ ] pending

**Intent:**
Encapsulate all key-file logic in a single dedicated module so it can be
tested independently and imported cleanly by the middleware.

**Expected Outcomes:**
- A new file `vllm_proxy/auth.py` exists.
- It exports one async function `load_keys(path: str) -> frozenset[str]` that
  reads the file and returns the set of non-empty, stripped lines.
- It raises `FileNotFoundError` / `PermissionError` as-is when the file cannot
  be read (callers handle those).

**Todo List:**
1. Create `vllm_proxy/auth.py`.
2. Implement `load_keys(path: str) -> frozenset[str]`:
   - Open the file in text mode, read all lines.
   - Strip whitespace; skip empty lines and lines starting with `#` (comments).
   - Return a `frozenset[str]` of the resulting tokens.
3. Add a module-level docstring explaining the expected file format
   (one key per line, `#` comments allowed, blank lines ignored).

**Relevant Context:**
- No existing auth module — create from scratch.
- Pattern mirrors the simple file-read already done in `load_config()` in
  `vllm_proxy/config.py`.

---

### Sub-Task 2 — Add FastAPI middleware in `vllm_proxy/app.py`

**Status:** [ ] pending

**Intent:**
Add a Starlette/FastAPI middleware function to `create_app()` that intercepts
every request, validates the `Authorization: Bearer <key>` header against the
key file, and short-circuits with the appropriate HTTP error when auth fails.

**Expected Outcomes:**
- All requests without a valid key are rejected before reaching any route.
- HTTP 503 is returned when `VLLM_KEYS_FILE` points to an unreadable / missing
  file.
- HTTP 401 is returned when the `Authorization` header is absent or malformed.
- HTTP 403 is returned when the key is present but not in the file.
- Valid requests pass through unchanged.
- The `VLLM_KEYS_FILE` env var is read once at app startup; if it is not set
  the server refuses to start (raises `RuntimeError`).

**Todo List:**
1. In `create_app()`, read `os.environ["VLLM_KEYS_FILE"]` (raise `RuntimeError`
   with a clear message if missing).
2. Store `keys_file_path` as a local variable accessible to the middleware
   closure (not on `app.state` — it never changes).
3. Register an `@app.middleware("http")` async function `auth_middleware`.
4. Inside `auth_middleware`:
   a. Call `load_keys(keys_file_path)` inside a `try/except (FileNotFoundError,
      PermissionError, OSError)` block — return `JSONResponse(503, ...)` on
      failure.
   b. Extract the `Authorization` header; if missing or not starting with
      `"Bearer "` return `JSONResponse(401, ...)`.
   c. Extract the token (everything after `"Bearer "`).
   d. If token not in the loaded key set return `JSONResponse(403, ...)`.
   e. Otherwise `await call_next(request)` and return the response.
5. Import `os` and `load_keys` from `vllm_proxy.auth` at the top of `app.py`.

**Relevant Context:**
- `vllm_proxy/app.py` — `create_app()` function (lines 37–188).
- FastAPI middleware pattern: `@app.middleware("http") async def name(request, call_next)`.
- Existing `app.state` stores `pool`, `config`, `http_client` — do not put
  `keys_file_path` there since it is immutable and only needed by the middleware.

---

### Sub-Task 3 — Validate startup behaviour in `__main__.py`

**Status:** [ ] pending

**Intent:**
Ensure the server refuses to start with a clear error message if
`VLLM_KEYS_FILE` is not set, so operators know exactly what is wrong.

**Expected Outcomes:**
- If `VLLM_KEYS_FILE` is not in the environment when `create_app()` is called,
  a `RuntimeError` propagates up.
- `__main__.py` catches it in the existing `try/except` block and exits with a
  meaningful log message.

**Todo List:**
1. In `__main__.py`, extend the `except` clause that already catches
   `(FileNotFoundError, ValueError)` to also catch `RuntimeError`.
2. No other changes needed — the error message originates in `create_app()`.

**Relevant Context:**
- `vllm_proxy/__main__.py` — lines 50–54 (existing exception handling block).

---

### Sub-Task 4 — Write tests in `tests/test_auth.py`

**Status:** [ ] pending

**Intent:**
Cover the new auth module and middleware with unit tests that follow the
existing test patterns in the project.

**Expected Outcomes:**
- `tests/test_auth.py` exists and all tests pass.
- `load_keys()` is tested for: normal file, empty file, comments, blank lines,
  missing file.
- The middleware is tested for: valid key → 200, missing header → 401, wrong
  key → 403, unreadable file → 503.
- Existing tests in `test_proxy.py` and `test_admin.py` continue to pass
  (they must be updated to set the `VLLM_KEYS_FILE` env var or patch it).

**Todo List:**
1. Create `tests/test_auth.py`.
2. Add `load_keys` unit tests using `tmp_path` (pytest fixture) to write
   temporary key files.
3. Add middleware integration tests using `TestClient`:
   - Patch `os.environ["VLLM_KEYS_FILE"]` to a temp file containing a known key.
   - Issue requests with and without the correct `Authorization` header.
   - Verify the response status codes match the spec above.
4. Update `tests/test_proxy.py` and `tests/test_admin.py`:
   - In each test that calls `TestClient`, set the `Authorization: Bearer <key>`
     header using a monkeypatched or pre-populated key file so they don't
     regress to 401/403.
   - The simplest approach: use `monkeypatch.setenv("VLLM_KEYS_FILE", ...)` or
     patch `vllm_proxy.auth.load_keys` to return a fixed set.

**Relevant Context:**
- `tests/test_proxy.py` and `tests/test_admin.py` — existing test structure
  and `make_config()` helper pattern.
- FastAPI `TestClient` supports passing `headers={"Authorization": "Bearer ..."}`.
- `_make_app_with_fake_pool()` helper in both test files constructs the app —
  the `VLLM_KEYS_FILE` env var must be set before `create_app()` is called.

---

### Sub-Task 5 — Update `DEPLOYMENT.md` and `config.yaml` comments

**Status:** [ ] pending

**Intent:**
Document the new requirement so operators know to set `VLLM_KEYS_FILE` before
starting the proxy and understand the expected file format.

**Expected Outcomes:**
- `DEPLOYMENT.md` has a section describing the `VLLM_KEYS_FILE` environment
  variable, the key file format, and the HTTP error behaviour.
- `config.yaml` has a comment noting that auth is controlled by `VLLM_KEYS_FILE`
  (not by the YAML file itself).

**Todo List:**
1. Add an "API Key Authorization" section to `DEPLOYMENT.md` covering:
   - `VLLM_KEYS_FILE` env var (required, absolute or relative path).
   - Key file format: one key per line, `#` comments ignored, blank lines ignored.
   - What happens when the file is missing/unreadable at request time (503).
   - How to add/remove keys (edit file; takes effect on the next request).
2. Add a brief comment to `config.yaml` noting auth is env-var-driven.

**Relevant Context:**
- `DEPLOYMENT.md` — existing deployment documentation.
- `config.yaml` — top-level configuration file.
