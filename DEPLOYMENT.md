# vllm-proxy — Deployment Guide

This document covers a single-node lab deployment of vllm-proxy on a Linux host
with one or more NVIDIA GPUs. The pattern uses a dedicated system user, a Python
virtual environment, a systemd service for process supervision, and logrotate for
log management.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [System User and Directory Layout](#2-system-user-and-directory-layout)
3. [Install the Proxy](#3-install-the-proxy)
4. [Configuration](#4-configuration)
5. [API Key Authorization](#5-api-key-authorization)
6. [systemd Service](#6-systemd-service)
7. [Log Rotation](#7-log-rotation)
8. [Firewall / Port Reference](#8-firewall--port-reference)
9. [Smoke Test](#9-smoke-test)
10. [Admin API Cheat Sheet](#10-admin-api-cheat-sheet)
11. [Updating the Code](#11-updating-the-code)
12. [Uninstall](#12-uninstall)

---

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 / RHEL 9 | Any systemd-based Linux |
| Python | 3.10 | 3.11+ recommended |
| VLLM | 0.4.x | Must be on `PATH` for the service user |
| NVIDIA driver | 525+ | Required by VLLM |
| CUDA toolkit | 12.x | Must match VLLM's build |
| Disk — models | Varies | Mounted at a stable path, e.g. `/models` |
| Disk — logs | ~10 GB | For VLLM per-model log files |
| Open ports | 8000, 9000–900N | Proxy port + one port per pool slot |

Verify VLLM is installed and reachable:

```bash
vllm --version
```

---

## 2. System User and Directory Layout

Run all commands as `root` or with `sudo`.

```bash
# Dedicated non-login service account
useradd --system --shell /usr/sbin/nologin --home /opt/vllm-proxy vllm-proxy

# Directory layout
mkdir -p /opt/vllm-proxy/{app,venv,logs}
chown -R vllm-proxy:vllm-proxy /opt/vllm-proxy
```

Resulting layout:

```
/opt/vllm-proxy/
├── app/          ← git checkout of this repo
├── venv/         ← Python virtual environment
├── logs/         ← VLLM per-model log files (managed by logrotate)
└── config.yaml   ← live configuration file
```

---

## 3. Install the Proxy

```bash
# Switch to the service account for all install steps
sudo -u vllm-proxy bash

# Clone the repo into /opt/vllm-proxy/app
git clone https://github.com/deadchannelsky/vllmrouter /opt/vllm-proxy/app

# Create the virtual environment
python3 -m venv /opt/vllm-proxy/venv

# Install the proxy and its dependencies
/opt/vllm-proxy/venv/bin/pip install --upgrade pip
/opt/vllm-proxy/venv/bin/pip install -r /opt/vllm-proxy/app/requirements.txt
/opt/vllm-proxy/venv/bin/pip install /opt/vllm-proxy/app

# Verify the entry point is available
/opt/vllm-proxy/venv/bin/vllm-proxy --help
```

---

## 4. Configuration

Copy the example config and edit it for your environment:

```bash
cp /opt/vllm-proxy/app/config.yaml /opt/vllm-proxy/config.yaml
chown vllm-proxy:vllm-proxy /opt/vllm-proxy/config.yaml
chmod 640 /opt/vllm-proxy/config.yaml
```

Edit `/opt/vllm-proxy/config.yaml`:

```yaml
host: "0.0.0.0"      # bind address for the proxy
port: 8000           # port callers connect to

log_dir: "/opt/vllm-proxy/logs"   # absolute path; must be writable by vllm-proxy user

pool:
  max_size: 2                     # max simultaneously loaded models
  base_port: 9000                 # VLLM instances get ports 9000, 9001, …
  startup_timeout_seconds: 180    # increase for large models (>30 GB)

models:
  mistral-7b:
    model_path: "/models/mistral-7b"
    vllm_args:
      - "--dtype=float16"
      - "--max-model-len=8192"
      - "--gpu-memory-utilization=0.90"

  llama3-8b:
    model_path: "/models/llama3-8b"
    vllm_args:
      - "--dtype=bfloat16"
      - "--tensor-parallel-size=2"
      - "--gpu-memory-utilization=0.90"
```

### Config field reference

| Field | Default | Description |
|---|---|---|
| `host` | `0.0.0.0` | Bind address for Uvicorn |
| `port` | `8000` | Proxy listener port |
| `log_dir` | `logs` | Directory for per-model VLLM logs |
| `pool.max_size` | — | Max simultaneously loaded models. Set to the number of GPUs (or GPU sets) you have available |
| `pool.base_port` | — | First port assigned to a VLLM instance. Ports `base_port` through `base_port + max_size - 1` must be free |
| `pool.startup_timeout_seconds` | `120` | How long to wait for a VLLM `/health` 200 before declaring startup failed |
| `models.<id>.model_path` | — | Absolute path to the model directory on disk |
| `models.<id>.vllm_args` | `[]` | Extra flags passed verbatim to `vllm serve` |

---

## 5. API Key Authorization

Every request to the proxy — including `/v1/*` and `/admin/*` endpoints — must
carry a valid API key in the standard OpenAI `Authorization` header:

```
Authorization: Bearer <api-key>
```

### How it works

The proxy reads the list of accepted keys from a plain-text file on disk.  The
path to that file is given by the **`VLLM_KEYS_FILE`** environment variable,
which must be set before the proxy starts.  The file is re-read on **every
request**, so keys added or removed by the external key-management application
take effect immediately — no restart required.

### Key file format

```
# Lines starting with '#' are comments and are ignored.
# Blank lines are also ignored.

sk-prod-abc123def456
sk-prod-xyz789ghi012

# CI / test key
sk-test-00000000
```

- One key per line.
- Leading and trailing whitespace is stripped.
- Lines starting with `#` are comments.
- Blank (whitespace-only) lines are ignored.
- The file may be empty; in that case all requests are denied.

### Setting up the key file

```bash
# Create the key file (owned and readable only by the service account)
touch /opt/vllm-proxy/keys.txt
chown vllm-proxy:vllm-proxy /opt/vllm-proxy/keys.txt
chmod 600 /opt/vllm-proxy/keys.txt

# Add your first key
echo "sk-prod-abc123def456" >> /opt/vllm-proxy/keys.txt
```

The external key-management application can append, remove, or rewrite keys in
this file at any time. The proxy will pick up the change on the very next
request.

### HTTP error responses

| Condition | Status |
|---|---|
| `Authorization` header missing or not in `Bearer <key>` format | `401 Unauthorized` |
| Key present but not in the key file | `403 Forbidden` |
| Key file is missing or unreadable at request time | `503 Service Unavailable` |

A `503` is returned (not `401`) when the file cannot be read, so operators can
distinguish an auth misconfiguration from a key-file I/O problem. The proxy
will recover automatically once the file becomes readable again.

### Startup guard

If `VLLM_KEYS_FILE` is not set in the environment, the proxy will **refuse to
start** and log a clear error:

```
CRITICAL Failed to start vllm-proxy: VLLM_KEYS_FILE environment variable is not set. ...
```

---

## 6. systemd Service

Create `/etc/systemd/system/vllm-proxy.service`:

```ini
[Unit]
Description=vllm-proxy — OpenAI-compatible LLM router
After=network.target
# If your models are on a network mount, add:
# After=network.target mnt-models.mount

[Service]
Type=simple
User=vllm-proxy
Group=vllm-proxy
WorkingDirectory=/opt/vllm-proxy

# Point to the installed entry point and config
ExecStart=/opt/vllm-proxy/venv/bin/vllm-proxy \
    --config /opt/vllm-proxy/config.yaml \
    --host 0.0.0.0 \
    --port 8000

# Required: path to the API key file managed by the external key application.
Environment=VLLM_KEYS_FILE=/opt/vllm-proxy/keys.txt

# Give VLLM processes time to flush GPU memory on shutdown.
# Must be greater than pool.startup_timeout_seconds to avoid
# the proxy being SIGKILL'd while waiting for children.
TimeoutStopSec=300

# Restart on crash, but not on a clean exit (e.g. config error on startup).
Restart=on-failure
RestartSec=10

# Ensure CUDA and the venv are visible
Environment=PATH=/opt/vllm-proxy/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Send stdout/stderr to journald
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vllm-proxy

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
systemctl daemon-reload
systemctl enable vllm-proxy
systemctl start vllm-proxy

# Verify it is running
systemctl status vllm-proxy

# Follow live logs
journalctl -u vllm-proxy -f
```

---

## 7. Log Rotation

VLLM writes verbose output. Without rotation a busy model's log will fill the
disk within days.

Create `/etc/logrotate.d/vllm-proxy`:

```
/opt/vllm-proxy/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

The `copytruncate` directive truncates the live file in place rather than moving
it, so the VLLM subprocess (which has the file descriptor open) keeps writing
without any signal or restart required.

Test the config:

```bash
logrotate --debug /etc/logrotate.d/vllm-proxy
```

---

## 8. Firewall / Port Reference

| Port | Direction | Purpose |
|---|---|---|
| `8000` | inbound | Proxy OpenAI API — open to lab clients |
| `9000`–`900N` | loopback only | VLLM subprocess instances — **do not expose externally** |

Example with `ufw`:

```bash
ufw allow 8000/tcp comment "vllm-proxy OpenAI API"
# VLLM ports are 127.0.0.1-bound by default — no rule needed
```

Example with `firewalld`:

```bash
firewall-cmd --permanent --add-port=8000/tcp
firewall-cmd --reload
```

---

## 9. Smoke Test

```bash
KEY="sk-your-key-here"

# 1. Confirm the proxy is up
curl -s http://localhost:8000/v1/models \
  -H "Authorization: Bearer $KEY" | python3 -m json.tool

# 2. Send a chat completion (triggers cold-start if model not yet warm)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "Hello, what are you?"}],
    "max_tokens": 64
  }' | python3 -m json.tool

# 3. Streaming response
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "Count to five."}],
    "max_tokens": 32,
    "stream": true
  }'

# 4. Inspect pool state via admin API
curl -s http://localhost:8000/admin/models \
  -H "Authorization: Bearer $KEY" | python3 -m json.tool
```

The first request to a cold model will block for up to
`startup_timeout_seconds` while VLLM loads. Subsequent requests return
immediately from the warm pool.

---

## 10. Admin API Cheat Sheet

All admin endpoints are on the same port as the proxy (`8000` by default) and
require the same `Authorization: Bearer <key>` header as the `/v1/*` routes.

```bash
BASE=http://localhost:8000
KEY="sk-your-key-here"

# List all configured models and which are warm
curl -s $BASE/admin/models \
  -H "Authorization: Bearer $KEY" | python3 -m json.tool

# Pre-warm a model immediately (blocks until ready or timeout)
curl -s -X POST $BASE/admin/models/mistral-7b/load \
  -H "Authorization: Bearer $KEY"

# Evict a model from the pool (frees GPU memory)
curl -s -X POST $BASE/admin/models/llama3-8b/unload \
  -H "Authorization: Bearer $KEY"

# Add a new model at runtime (ephemeral — lost on restart)
curl -s -X POST $BASE/admin/models/phi-3 \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/models/phi-3",
    "vllm_args": ["--dtype=float16"],
    "priority": 0
  }'

# Remove a model entirely (unloads first if warm)
curl -s -X DELETE $BASE/admin/models/phi-3 \
  -H "Authorization: Bearer $KEY"
```

---

## 11. Updating the Code

```bash
# Pull latest code
sudo -u vllm-proxy git -C /opt/vllm-proxy/app pull

# Re-install the package in the venv
sudo -u vllm-proxy /opt/vllm-proxy/venv/bin/pip install /opt/vllm-proxy/app

# Restart the service (graceful: waits for VLLM children to stop)
systemctl restart vllm-proxy
```

Config changes (editing `/opt/vllm-proxy/config.yaml`) also require a restart
because the config is loaded once at startup. Runtime model additions via the
admin API do not require a restart but are not persisted across restarts.

---

## 12. Uninstall

```bash
systemctl stop vllm-proxy
systemctl disable vllm-proxy
rm /etc/systemd/system/vllm-proxy.service
systemctl daemon-reload

rm -rf /opt/vllm-proxy
userdel vllm-proxy

rm /etc/logrotate.d/vllm-proxy
```
