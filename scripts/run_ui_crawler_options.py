"""
UI crawler focused on the Options tab.
Steps:
 - Launch Streamlit (same settings as run_ui_crawler).
 - Navigate to Options tab.
 - Enter ticker AAPL in the common ticker field.
 - Attempt to click "Ajouter au dashboard" buttons found in the Options page.
Errors are saved to logs/ui_error_options.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_ui_crawler as base_crawler  # noqa: E402

DEFAULT_PORT = os.getenv("STREAMLIT_OPTIONS_PORT", "8502")
STREAMLIT_URL = f"http://localhost:{DEFAULT_PORT}"
ERROR_JSON = Path("logs/ui_error_options.json")
STARTUP_WAIT = float(os.getenv("UI_STARTUP_WAIT", "6"))


def _start_streamlit_options() -> subprocess.Popen | None:
    app_path = base_crawler.STREAMLIT_APP
    if not app_path.exists():
        print(f"[!] Streamlit app introuvable: {app_path}")
        return None

    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env["STREAMLIT_SERVER_PORT"] = DEFAULT_PORT
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        env["STREAMLIT_SERVER_PORT"],
        "--server.headless",
        env["STREAMLIT_SERVER_HEADLESS"],
    ]

    print(f"[*] Demarrage Streamlit: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    if base_crawler._wait_for_streamlit_ready(STREAMLIT_URL):
        return proc

    print("[!] Streamlit ne repond pas, arret du processus.")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return None


def _save_error(message: str, error_type: str = "CrawlerError", location: str = "OptionsTab"):
    payload = {
        "location": location,
        "error_type": error_type,
        "message": message,
        "file": __file__,
        "line": 0,
        "traceback": "",
    }
    ERROR_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n[!!!] ERREUR SAUVEE DANS ui_error_options.json\n")
    print(json.dumps(payload, indent=2))


def _click_options_tab(page) -> bool:
    selectors = [
        '[data-testid="stTabs"] button[role="tab"]',
        '[data-testid="stTabs"] button',
        ".stTabs button",
        'div[role="tablist"] button[role="tab"]',
    ]
    for sel in selectors:
        buttons = page.locator(sel)
        count = buttons.count()
        for idx in range(count):
            label = buttons.nth(idx).inner_text().strip().lower()
            if "options" in label:
                buttons.nth(idx).click()
                return True
    return False


def _fill_ticker(page, ticker: str):
    try:
        page.wait_for_selector("input", timeout=5000)
    except Exception:
        pass
    # Prefer placeholder matching ex: AAPL
    try:
        inp = page.get_by_placeholder("ex: AAPL", exact=False)
        if inp.count() > 0:
            inp.first.fill(ticker)
            return True
    except Exception:
        pass
    try:
        inp = page.get_by_placeholder("AAPL", exact=False)
        if inp.count() > 0:
            inp.first.fill(ticker)
            return True
    except Exception:
        pass
    # Fallback to label containing "Ticker commun"
    try:
        inp = page.get_by_label("Ticker commun", exact=False)
        if inp.count() > 0:
            inp.first.fill(ticker)
            return True
    except Exception:
        pass
    # Generic fallback
    inputs = page.locator("input")
    for idx in range(inputs.count()):
        try:
            placeholder = inputs.nth(idx).get_attribute("placeholder") or ""
            label = inputs.nth(idx).get_attribute("aria-label") or ""
            if "ticker" in placeholder.lower() or "ticker" in label.lower():
                inputs.nth(idx).fill(ticker)
                return True
        except Exception:
            continue
    # Last resort: first text input
    try:
        if inputs.count() > 0:
            inputs.first.fill(ticker)
            return True
    except Exception:
        pass
    return False


def _click_add_buttons(page):
    btns = page.get_by_text("Ajouter au dashboard", exact=False)
    count = btns.count()
    clicked = 0
    for idx in range(count):
        try:
            btns.nth(idx).click()
            time.sleep(0.5)
            clicked += 1
        except Exception:
            continue
    return clicked


def run_ui_crawler_options():
    ERROR_JSON.parent.mkdir(parents=True, exist_ok=True)
    ERROR_JSON.write_text("{}", encoding="utf-8")

    streamlit_proc = _start_streamlit_options()
    if streamlit_proc is None:
        _save_error("Streamlit indisponible (Options crawler)", error_type="StartupError", location="OptionsTab")
        return

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(120000)

            print("[*] Ouverture Streamlit (Options)...")
            page.goto(STREAMLIT_URL)
            page.wait_for_load_state("networkidle")
            time.sleep(STARTUP_WAIT)

            err = base_crawler.extract_error(page.content())
            if err:
                _save_error(err.get("message", "Erreur au démarrage"), err.get("error_type", "StartupError"), "__startup__")
                browser.close()
                return

            if not _click_options_tab(page):
                _save_error("Onglet Options introuvable", error_type="TabError", location="OptionsTab")
                browser.close()
                return

            page.wait_for_timeout(2000)
            if not _fill_ticker(page, "AAPL"):
                _save_error("Champ ticker introuvable dans l'onglet Options", error_type="InputError", location="OptionsTab")
                browser.close()
                return

            # Laisser le temps aux closings de se charger
            time.sleep(2.0)
            clicked = _click_add_buttons(page)
            if clicked == 0:
                _save_error("Aucun bouton 'Ajouter au dashboard' cliqué", error_type="ActionError", location="OptionsTab")
            else:
                ok_payload = {"status": "ok", "actions": {"added": clicked}}
                ERROR_JSON.write_text(json.dumps(ok_payload, indent=2), encoding="utf-8")
                print(f"\n[V] ui_error_options.json mis à jour (ok, {clicked} clics).")

            browser.close()
    finally:
        if streamlit_proc:
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_proc.kill()


if __name__ == "__main__":
    run_ui_crawler_options()
