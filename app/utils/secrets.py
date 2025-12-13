from __future__ import annotations

import os
from pathlib import Path

_ENV_LOADED = False


def _load_dotenv_fallback() -> None:
    """
    Minimal `.env` loader used for local dev.

    Streamlit Community Cloud should use `st.secrets` (or environment variables).
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value
    except Exception:
        return


def get_secret(key: str, default: str | None = None) -> str | None:
    """
    Return configuration value from:
    1) OS env vars
    2) Streamlit secrets (if available)
    3) default
    """
    _load_dotenv_fallback()

    val = os.getenv(key)
    if val:
        return val

    try:
        import streamlit as st  # type: ignore

        try:
            secret_val = st.secrets.get(key)  # type: ignore[attr-defined]
        except Exception:
            secret_val = None
        if secret_val is not None and str(secret_val).strip() != "":
            return str(secret_val)
    except Exception:
        pass

    return default

