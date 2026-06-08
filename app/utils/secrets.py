from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

# MVC rule: app.utils must NOT depend on the view (no `import streamlit`).
# Streamlit's `st.secrets` is injected from the composition root (streamlit_app.py)
# via `set_secret_source`, so this module stays a pure, view-free ConfigProvider.

_ENV_LOADED = False
_SECRET_SOURCE: Optional[Callable[[str], Optional[str]]] = None


def set_secret_source(source: Optional[Callable[[str], Optional[str]]]) -> None:
    """
    Register a secondary secret source, consulted after OS env vars.

    The composition root (e.g. ``streamlit_app.py``) wires this to ``st.secrets``
    so the view layer never has to import ``app.utils`` and ``app.utils`` never
    imports Streamlit. Pass ``None`` to clear (useful in tests).
    """
    global _SECRET_SOURCE
    _SECRET_SOURCE = source


def _load_dotenv_fallback() -> None:
    """
    Minimal `.env` loader used for local dev.

    Production should inject secrets via `set_secret_source` (e.g. `st.secrets`)
    or plain environment variables.
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
    2) injected secret source (e.g. Streamlit secrets, wired at the composition root)
    3) default
    """
    _load_dotenv_fallback()

    val = os.getenv(key)
    if val:
        return val

    if _SECRET_SOURCE is not None:
        try:
            secret_val = _SECRET_SOURCE(key)
        except Exception:
            secret_val = None
        if secret_val is not None and str(secret_val).strip() != "":
            return str(secret_val)

    return default
