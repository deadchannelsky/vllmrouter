"""API key loader for vllm-proxy.

Reads accepted API keys from a plain-text file whose path is given by the
``VLLM_KEYS_FILE`` environment variable.

Key file format
---------------
- One key per line.
- Lines starting with ``#`` are treated as comments and ignored.
- Blank lines (or lines that are whitespace-only) are ignored.
- All other lines are stripped of leading/trailing whitespace and used as keys.

Example::

    # Production keys
    sk-prod-abc123
    sk-prod-def456

    # CI / test key
    sk-test-xyz789
"""

from __future__ import annotations


def load_keys(path: str) -> frozenset[str]:
    """Read *path* and return the set of valid API keys found inside.

    Parameters
    ----------
    path:
        Absolute or relative path to the key file.

    Returns
    -------
    frozenset[str]
        Immutable set of non-empty key strings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    PermissionError
        If the file cannot be read due to OS permissions.
    OSError
        For any other I/O failure.
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    keys: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            keys.add(stripped)

    return frozenset(keys)
