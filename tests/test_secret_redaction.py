"""redact() must mask shape-based tokens AND values injected via st.secrets.

Regression guard for the ultracode-verification findings (redact mirrored only 2
patterns and ignored the injected _SECRET_SOURCE). No network.
"""

from __future__ import annotations

import pytest

from app.utils.logging_config import redact
from app.utils import secrets as secrets_mod

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "token",
    [
        # Tokens assembled at runtime so no literal secret appears in source
        # (satisfies GitHub push protection + the repo secret scanner).
        "ghp_" + "a" * 36,
        "xoxb-" + "123456789012-" + "abcdEFGHijklMNOP",
        "AIza" + "B" * 35,
        "sk-" + "x" * 24,
        "sk_live_" + "y" * 24,
    ],
)
def test_redacts_shape_based_tokens(token):  # pragma: allowlist secret
    out = redact(f"leak {token} trailing")
    assert token not in out
    assert "REDACTED" in out


def test_redacts_unquoted_alpaca_secret_assignment():
    line = "APCA_API_SECRET_KEY=" + "Gb" * 20  # 40 chars, no PK/AK prefix  # pragma: allowlist secret
    out = redact(line)
    assert "REDACTED" in out


def test_redacts_value_injected_via_secret_source(monkeypatch):
    # The secret is available ONLY via the injected source (st.secrets), not os.environ.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(secrets_mod, "_ENV_LOADED", True, raising=False)
    secret_val = "totally-opaque-value-9f8e7d6c5b4a3210"  # pragma: allowlist secret
    secrets_mod.set_secret_source(lambda k: secret_val if k == "OPENAI_API_KEY" else None)
    try:
        out = redact(f"the resolved key is {secret_val} ok")
        assert secret_val not in out
        assert "REDACTED" in out
    finally:
        secrets_mod.set_secret_source(None)
