"""
ui_crawler_options_add_to_dashboard.py

OBJECTIF STRICT :
- Travailler UNIQUEMENT dans l’onglet "Options"
- Entrer "AAPL" dans le champ ticker
- Cliquer sur "Ajouter au dashboard"
- Logger toute erreur Streamlit (traceback HTML) dans logs/ui_error.json
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

# =========================
# CONFIG
# =========================

STREAMLIT_APP = Path("app/vue/main_app.py")
STREAMLIT_URL = "http://localhost:8501"
ERROR_JSON = Path("logs/ui_error.json")

STARTUP_WAIT = 6
PAGE_READY_TIMEOUT = 20
OPTIONS_TAB_NAME = "options"  # match case-insensitive

# =========================
# ERROR EXTRACTION
# =========================


def extract_streamlit_dom_error(html: str):
    error_type = re.search(r'<span[^>]*>([A-Za-z]+Error)</span>', html)
    error_type = error_type.group(1) if error_type else None

    msg = re.search(r'<div data-testid="stExceptionMessage">(.*?)</div>', html, re.DOTALL)
    if msg:
        msg = re.sub(r"<[^>]*>", "", msg.group(1)).strip()

    traceback_rows = re.findall(
        r'<div data-testid="stExceptionTraceRow".*?>(.*?)</div>', html, re.DOTALL
    )
    tb = "\n".join(re.sub(r"<[^>]*>", "", r).strip() for r in traceback_rows) if traceback_rows else None

    if not (error_type or msg or tb):
        return None

    file, line = "unknown", -1
    if tb:
        m = re.search(r'File "(.+?)", line (\d+)', tb)
        if m:
            file = m.group(1)
            line = int(m.group(2))

    return {
        "error_type": error_type or "UnknownError",
        "message": msg or "",
        "traceback": tb or "",
        "file": file,
        "line": line,
    }


def save_error(err: dict, location: str):
    payload = {
        "location": location,
        "error_type": err["error_type"],
        "message": err["message"],
        "file": err["file"],
        "line": err["line"],
        "traceback": err["traceback"],
    }
    ERROR_JSON.parent.mkdir(parents=True, exist_ok=True)
    ERROR_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[!!!] ERREUR SAUVEE DANS {ERROR_JSON}\n")
    print(json.dumps(payload, indent=2))


# =========================
# STREAMLIT BOOT
# =========================


def wait_for_streamlit_ready(url: str, timeout: int = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def start_streamlit():
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP),
    ]

    print("[*] Démarrage Streamlit")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    if not wait_for_streamlit_ready(STREAMLIT_URL):
        proc.terminate()
        raise RuntimeError("Streamlit ne démarre pas")

    return proc


def wait_for_page_ready(page, timeout=PAGE_READY_TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        html = page.content()
        if "stSpinner" not in html and "Running…" not in html:
            return True
        time.sleep(0.5)
    return False


# =========================
# UI ACTIONS — OPTIONS ONLY
# =========================


def switch_to_options_tab(page) -> bool:
    tabs = page.locator('button[role="tab"]')
    n = tabs.count()

    for i in range(n):
        label = tabs.nth(i).inner_text().strip().lower()
        if OPTIONS_TAB_NAME in label:
            print("[UI] Switch vers onglet Options")
            tabs.nth(i).click()
            time.sleep(2)

            err = extract_streamlit_dom_error(page.content())
            if err:
                save_error(err, "__options_tab__")
                return False

            return True

    save_error(
        {
            "error_type": "OptionsTabNotFound",
            "message": "Onglet 'Options' introuvable",
            "traceback": "",
            "file": "ui",
            "line": -1,
        },
        "__tabs__",
    )
    return False


def fill_ticker(page, ticker="AAPL") -> bool:
    inputs = page.locator('input[type="text"], input[type="search"]')
    n = inputs.count()

    for i in range(n):
        el = inputs.nth(i)
        placeholder = (el.get_attribute("placeholder") or "").lower()
        aria = (el.get_attribute("aria-label") or "").lower()

        if "ticker" in placeholder or "symbol" in placeholder or "ticker" in aria:
            print(f"[UI] Remplissage ticker = {ticker}")
            el.click()
            el.fill("")
            el.type(ticker, delay=30)
            el.press("Enter")
            time.sleep(1)

            err = extract_streamlit_dom_error(page.content())
            if err:
                save_error(err, "__ticker__")
                return False

            return True

    save_error(
        {
            "error_type": "TickerInputNotFound",
            "message": "Champ ticker introuvable dans l’onglet Options",
            "traceback": "",
            "file": "ui",
            "line": -1,
        },
        "__ticker__",
    )
    return False


def click_add_to_dashboard(page) -> bool:
    buttons = page.locator("button")
    n = buttons.count()

    for i in range(n):
        label = buttons.nth(i).inner_text().strip().lower()
        if "ajouter" in label and "dashboard" in label:
            print("[UI] Clic sur 'Ajouter au dashboard'")
            buttons.nth(i).click()
            time.sleep(2)

            err = extract_streamlit_dom_error(page.content())
            if err:
                save_error(err, "__add_to_dashboard__")
                return False

            return True

    save_error(
        {
            "error_type": "AddToDashboardButtonNotFound",
            "message": "Bouton 'Ajouter au dashboard' introuvable",
            "traceback": "",
            "file": "ui",
            "line": -1,
        },
        "__add_to_dashboard__",
    )
    return False


# =========================
# MAIN
# =========================


def main():
    ERROR_JSON.parent.mkdir(parents=True, exist_ok=True)
    ERROR_JSON.write_text("{}", encoding="utf-8")

    streamlit_proc = start_streamlit()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(STREAMLIT_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(STARTUP_WAIT)

        if not wait_for_page_ready(page):
            save_error(
                {
                    "error_type": "PageLoadTimeout",
                    "message": "La page ne s’est pas stabilisée",
                    "traceback": "",
                    "file": "ui",
                    "line": -1,
                },
                "__load__",
            )
            return

        if not switch_to_options_tab(page):
            return

        if not wait_for_page_ready(page):
            save_error(
                {
                    "error_type": "OptionsTabLoadTimeout",
                    "message": "L’onglet Options ne s’est pas stabilisé",
                    "traceback": "",
                    "file": "ui",
                    "line": -1,
                },
                "__options_tab__",
            )
            return

        if not fill_ticker(page, "AAPL"):
            return

        if not click_add_to_dashboard(page):
            return

        print("\n[V] Onglet Options : aucune erreur détectée")
        ERROR_JSON.write_text(json.dumps({"status": "ok"}, indent=2), encoding="utf-8")

        browser.close()

    streamlit_proc.terminate()


if __name__ == "__main__":
    main()
