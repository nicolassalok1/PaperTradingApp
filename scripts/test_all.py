"""
Unified validation script for PaperTradingApp.
Runs syntax checks, MVC rules, Alpaca connectivity, controller smoke tests,
RL inference sanity, and UI crawler verification.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

def step(msg: str):
    print(f"[STEP] {msg}", flush=True)


def fail(msg: str):
    print(f"[FAIL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run_cmd(cmd: List[str], cwd: Path | None = None):
    proc = subprocess.run(cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        fail(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def syntax_check():
    step("Syntax check (py_compile)")
    py_files = [Path(p) for p in run_cmd(["git", "ls-files", "*.py"]).splitlines() if p.strip()]
    for f in py_files:
        try:
            run_cmd([sys.executable, "-m", "py_compile", str(f)])
        except SystemExit:
            raise
        except Exception as exc:
            fail(f"Syntax/import error in {f}: {exc}")


def mvc_check():
    step("MVC import checks")
    py_files = [Path(p) for p in run_cmd(["git", "ls-files", "*.py"]).splitlines() if p.strip()]
    for f in py_files:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        is_model = "app/model/" in str(f).replace("\\", "/")
        is_controller = "app/controller/" in str(f).replace("\\", "/")
        is_view = "app/vue/" in str(f).replace("\\", "/")
        for line in text.splitlines():
            line_strip = line.strip()
            if line_strip.startswith("#"):
                continue
            if is_model:
                if "import streamlit" in line_strip or "from streamlit" in line_strip:
                    fail(f"MVC violation: model imports streamlit -> {f}")
                if "app.vue" in line_strip:
                    fail(f"MVC violation: model imports view -> {f}")
            if is_controller:
                if "app.vue" in line_strip:
                    fail(f"MVC violation: controller imports view -> {f}")
            if is_view:
                if "app.model" in line_strip and "controller" not in line_strip:
                    fail(f"MVC violation: view imports model directly -> {f}")


def alpaca_connectivity():
    step("Alpaca connectivity (read-only account summary)")
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL")

    if not key or not secret or not base:
        raise EnvironmentError(
            "Alpaca API keys missing: APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL must be set."
        )

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    url = base.rstrip("/") + "/v2/account"

    try:
        resp = requests.get(url, headers=headers, timeout=5)
    except Exception as e:
        raise RuntimeError(f"Alpaca connectivity failed (network error): {e}")

    if resp.status_code != 200:
        raise RuntimeError(
            f"Alpaca API returned non-200 status: {resp.status_code}. Body: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("Alpaca API returned invalid JSON.")

    required_keys = ["id", "status", "equity", "cash"]

    for k in required_keys:
        if k not in data:
            raise RuntimeError(f"Alpaca API JSON missing required field: {k}")

    print("[OK] Alpaca API connectivity verified: HTTP 200 and JSON valid")


def alpaca_positions_test():
    import os, requests

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL")

    if not key or not secret or not base:
        raise EnvironmentError("Alpaca credentials missing for positions test.")

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    url = base.rstrip("/") + "/v2/positions"
    resp = requests.get(url, headers=headers, timeout=5)

    if resp.status_code != 200:
        raise RuntimeError(
            f"/v2/positions returned {resp.status_code}. Body: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("Invalid JSON from /v2/positions")

    if not isinstance(data, list):
        raise RuntimeError("Expected a list of positions for /v2/positions")

    print("[OK] Alpaca /v2/positions validated.")


def alpaca_orders_test():
    import os, requests

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL")

    if not key or not secret or not base:
        raise EnvironmentError("Alpaca credentials missing for orders test.")

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    url = base.rstrip("/") + "/v2/orders"
    resp = requests.get(url, headers=headers, timeout=5)

    if resp.status_code != 200:
        raise RuntimeError(
            f"/v2/orders returned {resp.status_code}. Body: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("Invalid JSON from /v2/orders")

    if not isinstance(data, list):
        raise RuntimeError("Expected a list of orders from /v2/orders")

    print("[OK] Alpaca /v2/orders validated.")


def alpaca_activities_test():
    import os, requests

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL")

    if not key or not secret or not base:
        raise EnvironmentError("Alpaca credentials missing for activities test.")

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    url = base.rstrip("/") + "/v2/account/activities"

    resp = requests.get(url, headers=headers, timeout=5)

    if resp.status_code != 200:
        raise RuntimeError(
            f"/v2/account/activities returned {resp.status_code}. Body: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("Invalid JSON from /v2/account/activities")

    if not isinstance(data, list):
        raise RuntimeError("Expected a list of activities for /v2/account/activities")

    print("[OK] Alpaca /v2/account/activities validated.")


def alpaca_assets_test():
    import os, requests

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL")

    if not key or not secret or not base:
        raise EnvironmentError("Alpaca credentials missing for assets test.")

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    url = base.rstrip("/") + "/v2/assets"

    resp = requests.get(url, headers=headers, timeout=5)

    if resp.status_code != 200:
        raise RuntimeError(
            f"/v2/assets returned {resp.status_code}. Body: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("Invalid JSON from /v2/assets")

    if not isinstance(data, list):
        raise RuntimeError("Expected a list of assets for /v2/assets")

    print("[OK] Alpaca /v2/assets validated.")


def controllers_smoke():
    step("Controller smoke tests")
    from app.controller import dashboard_v2_controller as dash_ctrl
    from app.controller import hedger_v2_controller as hedger_ctrl
    from app.controller import rl_live_v2_controller as rl_ctrl
    from app.controller import portfolio_allocation_controller as pa_ctrl

    summary = dash_ctrl.get_account_summary()
    assert isinstance(summary, dict)
    _ = hedger_ctrl.get_equity_positions()
    _ = rl_ctrl.get_rl_suggestion("AAPL")
    _ = pa_ctrl.get_portfolio_snapshot()


def rl_inference_test():
    step("RL inference sanity test")
    from app.controller import rl_live_v2_controller as rl_ctrl

    suggestion = rl_ctrl.get_rl_suggestion("AAPL")
    action = suggestion.get("action", {})
    for key in ("action_id", "side", "delta_qty"):
        if key not in action:
            fail(f"RL action missing key: {key}")


def ui_crawler_test():
    step("UI crawler test")
    try:
        import playwright  # type: ignore
    except Exception:
        raise RuntimeError("Playwright is required for UI crawler tests but is not installed.")

    from scripts.run_ui_crawler import run_ui_crawler

    run_ui_crawler()
    for fname in ["ui_error.json", "ui_errors.json"]:
        fp = REPO_ROOT / fname
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if data:
                    fail(f"UI crawler detected errors in {fname}: {data}")
            except Exception:
                fail(f"Could not parse {fname}")


def main():
    syntax_check()
    mvc_check()
    alpaca_connectivity()
    alpaca_positions_test()
    alpaca_orders_test()
    alpaca_activities_test()
    alpaca_assets_test()
    controllers_smoke()
    rl_inference_test()
    ui_crawler_test()
    print("ALL TESTS PASSED - ENV STABLE - MVC CLEAN - NO ERRORS")


if __name__ == "__main__":
    main()
