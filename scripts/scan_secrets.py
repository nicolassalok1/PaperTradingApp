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

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Single source of truth for patterns (shared with app.utils.logging_config.redact).
from app.utils.secret_patterns import NAMED_PATTERNS, PLACEHOLDER

# Paths never scanned (the deliberate local secret mechanism).
ALLOWLIST_PREFIXES = (".env", ".streamlit/secrets.toml")
# Extensions we don't scan (binaries / generated).
SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".pyc", ".so", ".dll",
    ".zip", ".gz", ".parquet", ".pt", ".pth", ".safetensors", ".onnx", ".bin",
)
# Inline opt-out for intentional synthetic secrets (e.g. test positive-controls).
ALLOWLIST_COMMENT = "allowlist secret"


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
            if ALLOWLIST_COMMENT in line:
                continue  # intentional synthetic secret (test control), opted out
            for name, pat in NAMED_PATTERNS:
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
