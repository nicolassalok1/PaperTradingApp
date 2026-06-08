"""
Single source of truth for secret detection patterns (pure utils, no deps).

Used by BOTH app.utils.logging_config.redact (runtime log/prompt redaction) and
scripts/scan_secrets.py (tracked-file gate), so the two can never drift apart.
"""

from __future__ import annotations

import re

# Env var names whose resolved VALUES must never appear in logs/prompts/tracked files.
SECRET_ENV_KEYS = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "OPENAI_API_KEY",
)

# Shape-based detectors for common provider tokens.
TOKEN_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),               # OpenAI
    re.compile(r"sk_live_[0-9A-Za-z]{20,}"),             # Stripe live
    re.compile(r"\b[PA]K[A-Z0-9]{16,}\b"),               # Alpaca key id (PK/AK prefix)
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                 # AWS access key
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),              # GitHub PAT (classic)
    re.compile(r"\bgithub_pat_[0-9A-Za-z_]{60,}\b"),     # GitHub PAT (fine-grained)
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),     # Slack
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),           # Google API
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),  # PEM private key
]

# Assignment style; quotes OPTIONAL so env/yaml `KEY=<40 base64 chars>` (e.g. the
# prefix-less Alpaca APCA_API_SECRET_KEY) is also caught.
ASSIGNED_SECRET = re.compile(
    r"(?i)(api[_-]?key|secret[_-]?key|secret|token|password|apca_api_secret_key)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{20,}['\"]?"
)

# Looks secret-ish but is a placeholder/example.
PLACEHOLDER = re.compile(r"(?i)(your-|example|placeholder|xxx|dummy|<.*>|redacted)")

# All shape detectors as (name, regex) for reporting.
NAMED_PATTERNS = [(p.pattern[:18], p) for p in TOKEN_PATTERNS] + [
    ("assigned_secret", ASSIGNED_SECRET)
]
