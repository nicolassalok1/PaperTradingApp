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
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from scripts import run_ui_crawler as base_crawler

STREAMLIT_URL = os.getenv("STREAMLIT_URL", "http://localhost:8501")
ERROR_JSON = Path("logs/ui_error_options.json")
STARTUP_WAIT = float(os.getenv("UI_STARTUP_WAIT", "6"))


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
    # Prefer placeholder matching ex: AAPL
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

    streamlit_proc = base_crawler._start_streamlit()  # reuse existing bootstrap

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(120000)

        print("[*] Ouverture Streamlit (Options)...")
        page.goto(STREAMLIT_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(STARTUP_WAIT)

        if base_crawler.check_startup_error(page):
            browser.close()
            return

        if not _click_options_tab(page):
            _save_error("Onglet Options introuvable", error_type="TabError", location="OptionsTab")
            browser.close()
            return

        time.sleep(1.0)
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
        if streamlit_proc:
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_proc.kill()


if __name__ == "__main__":
    run_ui_crawler_options()
