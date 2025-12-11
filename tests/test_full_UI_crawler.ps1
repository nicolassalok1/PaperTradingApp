# ======================================================
# test_full_UI_crawler.ps1 — UI Error Scanner
# ======================================================

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== FULL UI CRAWLER STARTED ===" -ForegroundColor Cyan

# CONFIG
$port = 8501
$url = "http://localhost:$port"
$errorJson = Join-Path $repoRoot "ui_error.json"
$errorJsonForPy = $errorJson -replace "\\", "/"
$appPath = Join-Path $repoRoot "app/vue/main_app.py"
$tempFile = Join-Path $PSScriptRoot "ui_crawler_temp.py"

# Clean old error file
if (Test-Path $errorJson) { Remove-Item $errorJson }

# Launch Streamlit app
Write-Host "[*] Launching Streamlit..."
$process = Start-Process "streamlit" -ArgumentList "run", $appPath, "--server.port=$port" -WorkingDirectory $repoRoot -PassThru

Start-Sleep -Seconds 6  # wait app boot

# Python crawler script
$py = @"
import json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ERROR_FILE = Path("$errorJsonForPy")
STREAMLIT_URL = "$url"

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
"@

Set-Content $tempFile $py


Write-Host "[*] Running crawler..." -ForegroundColor Cyan
Push-Location $repoRoot
try {
    $output = python $tempFile 2>&1
} finally {
    Pop-Location
}

# Kill Streamlit process
if ($process) {
    try { Stop-Process -Id $process.Id -Force } catch { }
}

Remove-Item $tempFile -Force -ErrorAction SilentlyContinue

Write-Host "`n=== UI CRAWLER RESULT ===" -ForegroundColor Cyan

if ($output -match "<<FOUND_ERROR") {
    Write-Host "UI ERROR DETECTED — See ui_error.json" -ForegroundColor Red
} elseif ($output -match "<<NO_ERROR>>") {
    Write-Host "NO ERRORS FOUND" -ForegroundColor Green
} else {
    Write-Host "UNKNOWN STATE — check output:" -ForegroundColor Yellow
    Write-Host $output
}
