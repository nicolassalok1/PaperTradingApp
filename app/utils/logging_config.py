"""
Minimal redacted logging for PaperTradingApp (MVC: pure utils, no view dependency).

Goal: structured-ish, secret-safe logging at the config/adapters boundary so data
source, offline fallback, calibration failure, checkpoint load and (paper) order
submit/reject can be diagnosed WITHOUT ever leaking a secret value into a log line.

Usage:
    from app.utils.logging_config import get_logger
    log = get_logger(__name__)
    log.info("alpaca client init base=%s", base_url)   # secret values auto-redacted
"""

from __future__ import annotations

import logging

from app.utils.secret_patterns import (
    ASSIGNED_SECRET,
    SECRET_ENV_KEYS as _SECRET_ENV_KEYS,
    TOKEN_PATTERNS as _TOKEN_PATTERNS,
)

_REDACTION = "***REDACTED***"


def _resolved_secret(key: str):
    """Resolve a secret value via the same path the app uses (env + st.secrets).

    Imported lazily to avoid any import cycle and to pick up the injected source.
    """
    try:
        from app.utils.secrets import get_secret

        return get_secret(key)
    except Exception:
        return None


def redact(text: str) -> str:
    """Mask resolved secret values (env OR injected st.secrets) and secret-shaped tokens."""
    # 1) Exact resolved values — covers secrets injected via st.secrets, not just os.environ.
    for key in _SECRET_ENV_KEYS:
        val = _resolved_secret(key)
        if val and len(str(val)) >= 6 and str(val) in text:
            text = text.replace(str(val), _REDACTION)
    # 2) Shape-based detectors (single source of truth, shared with scan_secrets).
    for pat in _TOKEN_PATTERNS:
        text = pat.sub(_REDACTION, text)
    text = ASSIGNED_SECRET.sub(_REDACTION, text)
    return text


# Backwards-compatible private alias.
_redact = redact


class _RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            redacted = _redact(msg)
            if redacted != msg:
                record.msg = redacted
                record.args = ()
        except Exception:
            pass
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the redaction filter attached (idempotent)."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, _RedactionFilter) for f in logger.filters):
        logger.addFilter(_RedactionFilter())
    return logger
