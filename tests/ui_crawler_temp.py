import json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ERROR_FILE = Path("C:/Users/Nathalie Asus/Documents/papertradingapp/PaperTradingApp/logs/ui_error.json")
STREAMLIT_URL = "http://localhost:8501"

def extract_error(html: str):
    # Streamlit error blocks
    msg = re.findall(r'<div data-testid="stExceptionMessage">(.*?)</div>', html, re.DOTALL)
    tb = re.findall(r'<div data-testid="stExceptionTraceRow".*?>(.*?)</div>', html, re.DOTALL)
    if msg:
        clean_msg = re.sub(r"<[^>]*>", "", msg[0]).strip()
        clean_tb = [re.sub(r"<[^>]*>", "", x).strip() for x in tb]
        return {"message": clean_msg, "traceback": clean_tb}
    return None


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    page.goto(STREAMLIT_URL)
    time.sleep(5)  # wait tabs load

    html = page.content()
    err = extract_error(html)

    if err:
        ERROR_FILE.write_text(json.dumps(err, indent=2))
        print("<<FOUND_ERROR>>")
    else:
        print("<<NO_ERROR>>")

    # Try switching tabs
    try:
        tabs = page.query_selector_all("button[data-baseweb='tab']")
        for t in tabs:
            t.click()
            time.sleep(2)
            html = page.content()
            err = extract_error(html)
            if err:
                ERROR_FILE.write_text(json.dumps(err, indent=2))
                print("<<FOUND_ERROR_IN_TABS>>")
                break
    except:
        pass

    browser.close()
