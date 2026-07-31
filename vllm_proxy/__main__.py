"""CLI entry point for vllm-proxy.

Usage:
    python -m vllm_proxy --config config.yaml [--host 0.0.0.0] [--port 8000]
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vllm-proxy",
        description="OpenAI-compatible proxy in front of VLLM subprocess instances.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--host",
        default=None,
        metavar="HOST",
        help="Host to bind the proxy server to (overrides config).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="Port to bind the proxy server to (overrides config).",
    )
    args = parser.parse_args()

    # Configure root logging before importing app so all modules share the format.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from vllm_proxy.config import load_config, scan_and_register_models
    from vllm_proxy.app import create_app

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logging.critical("Failed to load config: %s", exc)
        sys.exit(1)

    new_models = scan_and_register_models(config, args.config)
    if new_models:
        logging.info("Auto-registered %d model(s) from scan paths: %s", len(new_models), new_models)

    try:
        app = create_app(config)
    except RuntimeError as exc:
        logging.critical("Failed to start vllm-proxy: %s", exc)
        sys.exit(1)

    # CLI flags override config file values.
    host = args.host or config.host
    port = args.port or config.port

    import uvicorn

    logging.info(
        "Starting vllm-proxy on http://%s:%d  |  pool_max=%d  |  models=%d",
        host,
        port,
        config.pool.max_size,
        len(config.models),
    )
    uvicorn.run(app, host=host, port=port, log_config=None)


if __name__ == "__main__":
    main()
