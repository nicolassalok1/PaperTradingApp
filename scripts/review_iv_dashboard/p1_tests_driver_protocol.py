"""
Probe: robustness of the stdout-marker protocol used by
tests/integration/test_iv_dashboard_render.py (parser copied verbatim) and the
Windows encoding path of the subprocess (PYTHONIOENCODING / encoding='utf-8').
No network.
"""
import json
import os
import subprocess
import sys

_RESULT_MARKER = "IVDASH_RESULT "


def parse(stdout: str):
    """Verbatim copy of the fixture's parsing logic."""
    return next(
        (ln[len(_RESULT_MARKER):] for ln in stdout.splitlines() if ln.startswith(_RESULT_MARKER)),
        None,
    )


cases = {
    "warnings before marker": "some warning\nanother line\n" + _RESULT_MARKER + json.dumps({"a": 1}),
    "marker twice (first wins)": _RESULT_MARKER + json.dumps({"first": True}) + "\n" + _RESULT_MARKER + json.dumps({"second": True}),
    "marker string inside JSON value": _RESULT_MARKER + json.dumps({"exceptions": ["x IVDASH_RESULT y"]}),
    "exception text with newline (json escapes it)": _RESULT_MARKER + json.dumps({"exceptions": ["line1\nline2"]}),
    "marker mid-line (not at col 0)": "prefix " + _RESULT_MARKER + json.dumps({"a": 1}),
    "non-ascii in payload (ensure_ascii default)": _RESULT_MARKER + json.dumps({"t": "🌡️ Vol Implicite — échéance"}),
    "CRLF line endings": _RESULT_MARKER + json.dumps({"a": 1}) + "\r\n",
}
for name, out in cases.items():
    line = parse(out)
    try:
        val = json.loads(line) if line is not None else None
        status = f"parsed={val}"
    except Exception as exc:  # noqa: BLE001
        status = f"JSON ERROR {exc}"
    print(f"[{name:48s}] line_found={line is not None} {status}")
print("ensure_ascii keeps result line pure ASCII:", json.dumps({"t": "🌡️ é"}).isascii())

# ---- encoding of the child on Windows ---------------------------------------------
child = "import sys; print('IVDASH_RESULT {\"t\": \"🌡️ échéance\"}'); print('🌡️ err', file=sys.stderr)"
base_env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
for label, env in (
    ("with PYTHONIOENCODING=utf-8 (as fixture)", {**base_env, "PYTHONIOENCODING": "utf-8"}),
    ("without PYTHONIOENCODING", base_env),
):
    c = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, timeout=60)
    print(f"[{label}] rc={c.returncode} stdout={c.stdout.strip()!r} stderr_tail={c.stderr.strip().splitlines()[-1] if c.stderr.strip() else ''!r}")

# ---- returncode not asserted: child prints the marker then dies ----------------------
child2 = "print('IVDASH_RESULT {\"seeded\": {\"exceptions\": [], \"n_charts\": 3, \"n_metrics\": 3}, \"empty\": {\"exceptions\": [], \"has_info\": true, \"n_charts\": 0}}'); import sys; sys.exit(3)"
c = subprocess.run([sys.executable, "-c", child2], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
print(f"[marker then exit 3] rc={c.returncode} parsed={parse(c.stdout) is not None} -> fixture would pass (rc never checked)")
