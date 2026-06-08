"""
Dependency-free secret scanner — the one-time net from PLAN.md step 1 (gitleaks
substitute) and the anti-secret guard for tracked files / tests fixtures (step 4).

Scans git-TRACKED text files for secret-looking tokens. Local gitignored stores
(`.env`, `.streamlit/secrets.toml`) are intentionally NOT scanned (they are the
chosen local secret mechanism and never tracked). Binary/large files are skipped.

Exit 0 = clean, 1 = potential secret found.
Run: python scripts/scan_secrets.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Paths never scanned (the deliberate local secret mechanism).
ALLOWLIST_PREFIXES = (".env", ".streamlit/secrets.toml")
# Extensions we don't scan (binaries / generated).
SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".pyc", ".so", ".dll",
    ".zip", ".gz", ".parquet", ".pt", ".pth", ".safetensors", ".onnx", ".bin",
)

PATTERNS = {
    "openai_key": re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    "alpaca_key": re.compile(r"\b[PA]K[A-Z0-9]{16,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_pat": re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    "assigned_secret": re.compile(
        r"(?i)(api[_-]?key|secret[_-]?key|secret|token|password)\s*[:=]\s*"
        r"['\"][A-Za-z0-9_\-]{20,}['\"]"
    ),
}
# Tokens that look secret-ish but are placeholders/examples.
PLACEHOLDER = re.compile(r"(?i)(your-|example|placeholder|xxx|dummy|<.*>|redacted)")


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def is_allowlisted(rel: str) -> bool:
    return any(rel.replace("\\", "/").startswith(p) for p in ALLOWLIST_PREFIXES)


def scan() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for f in tracked_files():
        rel = str(f.relative_to(REPO_ROOT))
        if is_allowlisted(rel) or f.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name, pat in PATTERNS.items():
                m = pat.search(line)
                if m and not PLACEHOLDER.search(m.group(0)):
                    hits.append((rel, i, f"{name}: {m.group(0)[:12]}..."))
    return hits


def main() -> None:
    hits = scan()
    if not hits:
        print("[secret-scan] OK — no secrets found in tracked files.")
        sys.exit(0)
    print("[secret-scan] POTENTIAL SECRETS DETECTED:")
    for rel, line, what in hits:
        print(f" - {rel}:{line} {what}")
    sys.exit(1)


if __name__ == "__main__":
    main()
