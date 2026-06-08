"""
Anti-secret guard for the test fixtures.

Scans every file under tests/fixtures/ and fails if any value matches a known
secret pattern (OpenAI sk-, Alpaca PK/AK, AWS AKIA, GitHub ghp_, or an
`api_key/secret/token/password = "<long value>"` assignment).

Oracle independence: the detection regexes are NOT redefined here — they are
imported from the production scanner (scripts/scan_secrets.py), the same logic
that guards the whole tracked tree. Positive-control tests below feed synthetic
secret strings to that imported logic to prove the detector is not vacuously
green (it really fires on a secret), then we assert the real fixtures are clean.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


# Detection regexes come from the SINGLE source of truth shared by both the
# production scanner (scripts/scan_secrets.py) and runtime redaction
# (app.utils.logging_config.redact) — so this test can never drift from either.
from app.utils.secret_patterns import NAMED_PATTERNS, PLACEHOLDER


def _scan_text(text: str) -> list[tuple[str, str]]:
    """Return [(pattern_name, matched_token)] for real (non-placeholder) hits."""
    hits: list[tuple[str, str]] = []
    for line in text.splitlines():
        for name, pat in NAMED_PATTERNS:
            m = pat.search(line)
            if m and not PLACEHOLDER.search(m.group(0)):
                hits.append((name, m.group(0)))
    return hits


def _fixture_files() -> list[Path]:
    return [p for p in FIXTURES_DIR.rglob("*") if p.is_file()]


# --- Sanity: fixtures actually exist (guards against a vacuous pass) ----------

def test_fixtures_dir_is_populated():
    files = _fixture_files()
    assert files, f"no fixture files found under {FIXTURES_DIR}"
    # The brief requires a redacted Yahoo and Alpaca sample.
    names = {p.name.lower() for p in files}
    assert any("yahoo" in n for n in names), f"missing a yahoo fixture in {names}"
    assert any("alpaca" in n for n in names), f"missing an alpaca fixture in {names}"


# --- Positive controls: prove the imported detector really fires --------------

@pytest.mark.parametrize(
    "name, payload",
    [
        # Tokens assembled at runtime so no literal secret appears in source
        # (satisfies GitHub push protection + the repo secret scanner).
        ("openai_key", "sk-" + "abcdEFGH1234567890ZZZZ1234"),
        ("alpaca_key", "PK" + "ABCDEFGHIJ1234567890"),
        ("alpaca_secret", "AK" + "ABCDEFGHIJ1234567890ZZZZ"),
        ("aws_key", "AKIA" + "Z3K7Q9W1N5R2T8P4"),  # AKIA + exactly 16 [0-9A-Z]
        ("github_pat", "ghp_" + "a" * 36),
        ("assigned_token", 'token = "' + "abcdefghijklmnopqrstuvwxyz123456" + '"'),
        ("assigned_apikey", 'api_key: "' + "abcdefghijklmnopqrstuvwxyz123456" + '"'),
    ],
)
def test_detector_fires_on_synthetic_secret(name, payload):
    hits = _scan_text(payload)
    assert hits, f"detector failed to flag synthetic secret [{name}]: {payload!r}"


def test_detector_ignores_placeholder():
    # A token shaped like a secret but obviously a placeholder must NOT trip.
    assert _scan_text('api_key = "your-api-key-placeholder-xxxx"') == []


# --- The real guard: every fixture must be clean ------------------------------

@pytest.mark.parametrize(
    "fixture",
    _fixture_files(),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_fixture_contains_no_secret(fixture: Path):
    text = fixture.read_text(encoding="utf-8", errors="ignore")
    hits = _scan_text(text)
    assert not hits, (
        f"potential secret in fixture {fixture.relative_to(REPO_ROOT)}: "
        + ", ".join(f"{n}={tok[:12]}..." for n, tok in hits)
    )
