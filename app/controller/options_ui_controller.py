"""UI-facing controller helpers for options pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.controller.options_controller import (
    _get_cached_iv_for as _options_get_cached_iv_for,
    get_json_dir as _options_get_json_dir,
)


def get_cached_iv_for(*args: Any, **kwargs: Any):
    """Proxy cached IV lookup to the options controller."""
    return _options_get_cached_iv_for(*args, **kwargs)


def get_json_dir() -> Path:
    """Expose the JSON directory for dashboard cache interactions."""
    return _options_get_json_dir()


__all__ = ["get_cached_iv_for", "get_json_dir"]
