import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
except ModuleNotFoundError:
    print("[ERROR] Playwright is not installed. Run: pip install playwright && playwright install")
    raise


# =========================
# CONFIG
# =========================
ERROR_JSON = Path("logs/ui_error.json")
STREAMLIT_URL = "http://localhost:8501"
STREAMLIT_APP = Path("app/vue/main_app.py")

OPTIONS_TOP_TAB_NAME = "Options"
COMMON_TICKER = "AAPL"

STARTUP_WAIT = 6
TAB_WAIT = 4
TAB_DISCOVERY_TIMEOUT = 25
GRAPH_WAIT_TIMEOUT = 20
OPTION_FAMILIES = [
    "Vanilla / Early Exercise",
    "Path-dependent",
    "BarriŠres",
    "Spreads & Wings",
    "Calendriers",
    "Exotiques avanc‚es",
    "Basket",
]


# =========================
# ERROR EXTRACTION
# =========================
def extract_streamlit_dom_error(html: str):
    if 'data-testid="stException"' not in html:
        return None

    error_type = re.search(r'([A-Za-z]+Error)', html)
    error_type = error_type.group(1) if error_type else "UnknownError"

    msg_match = re.search(
        r'<div data-testid="stExceptionMessage">(.*?)</div>', html, re.DOTALL
    )
    message = (
        re.sub(r"<[^>]*>", "", msg_match.group(1)).strip()
        if msg_match
        else ""
    )

    tb_rows = re.findall(
        r'<div data-testid="stExceptionTraceRow".*?>(.*?)</div>', html, re.DOTALL
    )
    traceback = "\n".join(re.sub(r"<[^>]*>", "", r).strip() for r in tb_rows)

    file, line = "unknown", -1
    m = re.search(r'File "(.+?)", line (\d+)', traceback)
    if m:
        file, line = m.group(1), int(m.group(2))

    return {
        "error_type": error_type,
        "message": message,
        "traceback": traceback,
        "file": file,
        "line": line,
    }


def extract_error(html: str):
    return extract_streamlit_dom_error(html)


# =========================
# STREAMLIT START
# =========================
def _wait_for_streamlit_ready(url: str, timeout: int = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=3).status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def _start_streamlit():
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(STREAMLIT_APP),
        "--server.headless", "true",
        "--server.port", "8501",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    if _wait_for_streamlit_ready(STREAMLIT_URL):
        return proc

    proc.terminate()
    return None


# =========================
# LOGGING
# =========================
def save_error(err: dict, location: str):
    payload = {"location": location, **err}
    ERROR_JSON.parent.mkdir(parents=True, exist_ok=True)
    ERROR_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit("[UI ERROR DETECTED]")


def write_ok():
    ERROR_JSON.parent.mkdir(parents=True, exist_ok=True)
    ERROR_JSON.write_text(json.dumps({"status": "ok"}, indent=2), encoding="utf-8")
    print("[OK] Aucun problème détecté.")


# =========================
# UI HELPERS
# =========================
def safe_text(s: str) -> str:
    return s.encode("cp1252", errors="ignore").decode("cp1252").strip() if s else ""


def click_top_tab_options(page):
    tabs = page.get_by_role("tab")
    for i in range(tabs.count()):
        if safe_text(tabs.nth(i).inner_text()).lower() == OPTIONS_TOP_TAB_NAME.lower():
            tabs.nth(i).click()
            time.sleep(TAB_WAIT)
            return
    save_error(
        {
            "error_type": "UIError",
            "message": "Onglet Options introuvable",
            "traceback": "",
            "file": "unknown",
            "line": -1,
        },
        "__navigation__",
    )


def fill_common_ticker(page, ticker: str):
    selectors = [
        'input[aria-label="Ticker commun pour les historiques IV/clôtures (optionnel)"]',
        'label:has-text("Ticker commun") >> xpath=..//input',
        'input[type="text"]',
    ]

    input_box = None
    for sel in selectors:
        loc = page.locator(sel)
        if loc.count() == 1:
            input_box = loc.first
            break

    if not input_box:
        save_error(
            {
                "error_type": "UIError",
                "message": "Champ ticker commun introuvable",
                "traceback": "",
                "file": "unknown",
                "line": -1,
            },
            "__ticker__",
        )

    input_box.click()
    input_box.fill("")
    time.sleep(0.2)
    input_box.fill(ticker)
    input_box.press("Enter")
    print(f"[+] Ticker {ticker} injecté")


def wait_for_graph(page, timeout: int = GRAPH_WAIT_TIMEOUT):
    """
    Attends l'apparition d'un graphique Plotly ou Altair.
    """
    graph_selectors = [
        "div.plotly-graph-div",        # Plotly
        "div[data-testid='stPlotlyChart']",
        "div.vega-embed",              # Altair
        "canvas",                      # fallback (Altair canvas)
    ]

    start = time.time()
    while time.time() - start < timeout:
        for sel in graph_selectors:
            if page.locator(sel).count() > 0:
                print(f"[+] Graph détecté via selector: {sel}")
                return
        time.sleep(1)

    save_error(
        {
            "error_type": "UIError",
            "message": "Aucun graphique Plotly/Altair détecté après chargement",
            "traceback": "",
            "file": "unknown",
            "line": -1,
        },
        "__graph__",
    )


def discover_option_subtabs(page):
    tabs = page.get_by_role("tab")
    names = []
    for i in range(tabs.count()):
        name = safe_text(tabs.nth(i).inner_text())
        if name and name.lower() != OPTIONS_TOP_TAB_NAME.lower():
            names.append(name)
    found = list(dict.fromkeys(names))
    filtered = [n for n in found if n in OPTION_FAMILIES]
    return filtered or found


def crawl_option_tabs(page):
    subtabs = discover_option_subtabs(page)
    if not subtabs:
        save_error(
            {
                "error_type": "UIError",
                "message": "Aucun sous-onglet Options détecté",
                "traceback": "",
                "file": "unknown",
                "line": -1,
            },
            "__options__",
        )

    print(f"[+] Sous-onglets Options: {subtabs}")

    for name in subtabs:
        print(f"\n--- Onglet Options -> {name} ---")
        tab_locator = page.get_by_role("tab")
        clicked = False
        for i in range(tab_locator.count()):
            label = safe_text(tab_locator.nth(i).inner_text()).strip()
            if label.lower() == name.lower() or name.lower() in label.lower() or label.lower() in name.lower():
                tab_locator.nth(i).click()
                clicked = True
                break
        if not clicked:
            try:
                page.get_by_text(name, exact=False).first.click()
                clicked = True
            except Exception:
                pass
        if not clicked:
            labels = []
            for i in range(tab_locator.count()):
                labels.append(safe_text(tab_locator.nth(i).inner_text()))
            print(f"[WARN] Tab labels available: {labels}")
            save_error(
                {
                    "error_type": "UIError",
                    "message": f"Impossible de cliquer sur l'onglet {name}",
                    "traceback": "",
                    "file": "unknown",
                    "line": -1,
                },
                "__options_click__",
            )
        time.sleep(TAB_WAIT)

        wait_for_graph(page)

        err = extract_error(page.content())
        if err:
            save_error(err, name)


# =========================
# MAIN
# =========================
def main():
    ERROR_JSON.write_text("{}", encoding="utf-8")

    streamlit_proc = _start_streamlit()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(STREAMLIT_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(STARTUP_WAIT)

        click_top_tab_options(page)
        fill_common_ticker(page, COMMON_TICKER)
        wait_for_graph(page)

        crawl_option_tabs(page)

        write_ok()
        browser.close()

    if streamlit_proc:
        streamlit_proc.terminate()


if __name__ == "__main__":
    main()
