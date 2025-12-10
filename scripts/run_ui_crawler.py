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

ERROR_JSON = Path("logs/ui_error.json")
STREAMLIT_URL = "http://localhost:8506"
STREAMLIT_APP = Path("app/vue/main_app.py")

STARTUP_WAIT = 6
TAB_WAIT = 4
TAB_DISCOVERY_TIMEOUT = 20


def extract_streamlit_dom_error(html: str):
    """Parse Streamlit exception blocks using data-testid markup."""
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

    file, line = None, None
    if tb:
        m = re.search(r'File "(.+?)", line (\d+)', tb)
        if m:
            file = m.group(1)
            line = int(m.group(2))

    return {
        "error_type": error_type or "UnknownError",
        "message": msg or "",
        "traceback": tb or "",
        "file": file or "unknown",
        "line": line or -1,
    }


def extract_generic_traceback(html: str):
    """Fallback regex for raw Python traceback."""
    m = re.search(
        r"(Traceback \(most recent call last\:[\s\S]+?)([A-Za-z]+Error\: .+)", html, re.IGNORECASE
    )
    if not m:
        return None

    full = m.group(1) + m.group(2)
    clean = re.sub(r"<[^>]*>", "", full)
    clean = clean.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&").strip()

    fm = re.search(r'File "(.+?)", line (\d+)', clean)
    file = fm.group(1) if fm else "unknown"
    line = int(fm.group(2)) if fm else -1

    em = re.search(r"([A-Za-z]+Error)", clean)
    etype = em.group(1) if em else "UnknownError"

    mm = re.search(r"[A-Za-z]+Error: (.+)", clean)
    msg = mm.group(1).strip() if mm else clean[:200]

    return {
        "error_type": etype,
        "message": msg,
        "traceback": clean,
        "file": file,
        "line": line,
    }


def extract_minimal_error(html: str):
    """Last-resort extraction: first line containing 'Error:'."""
    m = re.search(r"([A-Za-z]+Error)\: (.+)", html)
    if not m:
        return None

    etype = m.group(1)
    msg = m.group(2).strip()

    clean_html = re.sub(r"<[^>]*>", "", html)
    clean_html = clean_html.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")

    fm = re.search(r'File "(.+?)", line (\d+)', clean_html)
    file = fm.group(1) if fm else "unknown"
    line = int(fm.group(2)) if fm else -1

    return {
        "error_type": etype,
        "message": msg,
        "traceback": clean_html[:2000],
        "file": file,
        "line": line,
    }


def extract_error(html: str):
    for extractor in (extract_streamlit_dom_error, extract_generic_traceback, extract_minimal_error):
        err = extractor(html)
        if err:
            return err
    return None


def _wait_for_streamlit_ready(url: str, timeout: int = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def _start_streamlit() -> subprocess.Popen | None:
    if not STREAMLIT_APP.exists():
        print(f"[!] Streamlit app introuvable: {STREAMLIT_APP}")
        return None

    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_SERVER_PORT", "8501")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP),
        "--server.port",
        env["STREAMLIT_SERVER_PORT"],
        "--server.headless",
        env["STREAMLIT_SERVER_HEADLESS"],
    ]

    print(f"[*] Demarrage Streamlit: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=Path(".").resolve(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    if _wait_for_streamlit_ready(STREAMLIT_URL):
        return proc

    print("[!] Streamlit ne repond pas, arret du processus.")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return None


def save_error_structured(err: dict, location: str):
    payload = {
        "location": location,
        "error_type": err["error_type"],
        "message": err["message"],
        "file": err["file"],
        "line": err["line"],
        "traceback": err["traceback"],
    }
    ERROR_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n[!!!] ERREUR SAUVEE DANS ui_error.json\n")
    print(json.dumps(payload, indent=2))


def check_startup_error(page) -> bool:
    html = page.content()
    err = extract_error(html)
    if err:
        save_error_structured(err, "__startup__")
        return True
    return False


def check_tabs(page) -> bool:
    selectors = [
        '[data-testid="stTabs"] button[role="tab"]',
        '[data-testid="stTabs"] button',
        ".stTabs button",
        'div[role="tablist"] button[role="tab"]',
    ]

    tabs = None
    used_selector = None
    start = time.time()
    while time.time() - start < TAB_DISCOVERY_TIMEOUT:
        for sel in selectors:
            candidate = page.locator(sel)
            if candidate.count() > 0:
                tabs = candidate
                used_selector = sel
                break
        if tabs:
            break
        time.sleep(1)

    if not tabs:
        print("[!] Aucun onglet detecte apres attente.")
        return False

    count = tabs.count()
    print(f"[+] {count} onglets detectes via '{used_selector}'.")

    for i in range(count):
        tab_name = tabs.nth(i).inner_text()
        safe_name = tab_name.encode("cp1252", errors="ignore").decode("cp1252")
        print(f"\n--- Switch vers onglet : {safe_name} ---")
        try:
            tabs.nth(i).click()
        except PlaywrightTimeoutError:
            print(f"[!] Timeout lors du clic sur l'onglet {safe_name}, on continue.")
            continue
        time.sleep(TAB_WAIT)

        html = page.content()
        err = extract_error(html)
        if err:
            save_error_structured(err, tab_name)
            return True

    print("\n[V] Aucun probleme detecte dans les onglets.")
    return False


def main():
    ERROR_JSON.parent.mkdir(parents=True, exist_ok=True)
    ERROR_JSON.write_text("{}", encoding="utf-8")

    streamlit_proc = _start_streamlit()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(120000)

        print("[*] Ouverture Streamlit...")
        page.goto(STREAMLIT_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(STARTUP_WAIT)

        if check_startup_error(page):
            browser.close()
            return

        had_error = check_tabs(page)
        if not had_error:
            ok_payload = {"status": "ok"}
            ERROR_JSON.write_text(json.dumps(ok_payload, indent=2), encoding="utf-8")
            print("\n[V] ui_error.json mis a jour (pas d'erreur).")

        browser.close()
        if streamlit_proc:
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_proc.kill()


def run_ui_crawler():
    main()


if __name__ == "__main__":
    main()
