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
import os
import re

# Env var names whose VALUES must never appear in logs.
_SECRET_ENV_KEYS = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "OPENAI_API_KEY",
)

# Generic secret-looking token patterns (OpenAI sk-, Alpaca PK.../AK..., long hex).
_TOKEN_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b[PA]K[A-Z0-9]{12,}\b"),
]

_REDACTION = "***REDACTED***"


def redact(text: str) -> str:
    """Mask known secret env values and secret-looking tokens in a string."""
    for key in _SECRET_ENV_KEYS:
        val = os.getenv(key)
        if val and len(val) >= 6 and val in text:
            text = text.replace(val, _REDACTION)
    for pat in _TOKEN_PATTERNS:
        text = pat.sub(_REDACTION, text)
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
