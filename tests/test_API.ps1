# ================================
# test_API.ps1 — Full Market/Options API Test
# ================================

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$scriptName = [IO.Path]::GetFileNameWithoutExtension($PSCommandPath)
$logFile = Join-Path $logsDir "$scriptName.log"
$env:PYTHONPATH = $repoRoot

$startedTranscript = $false
if (-not (Get-Variable -Name "__test_transcript_active" -Scope Global -ErrorAction SilentlyContinue)) {
    try {
        Start-Transcript -Path $logFile -Append -Force | Out-Null
        $global:__test_transcript_active = $true
        $startedTranscript = $true
    } catch {
        Write-Warning ("Transcript start failed for {0}: {1}" -f $scriptName, $_)
    }
}

# Colors
function Write-OK($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-FAIL($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red }

Write-Host "=== Running Market & Options API Tests ===" -ForegroundColor Cyan

# Generate a temporary Python test file
$py = @"
import traceback
from datetime import datetime

results = {}

def safe(name, fn):
    try:
        out = fn()
        results[name] = ("OK", out)
    except Exception as e:
        results[name] = ("FAIL", f"{type(e).__name__}: {e}")
        traceback.print_exc()

# ============================
# IMPORT TEST
# ============================

def test_imports():
    import app.model.market_data.market_data as md
    import app.model.options.logic as opt_logic
    import app.model.options.data.iv_surface as ivs
    import app.model.options.mc_engine as mc
    import app.model.options.context as ctx
    return "imports successful"

safe("imports", test_imports)


# ============================
# SPOT TEST
# ============================

def test_spot():
    from app.model.market_data.market_data import fetch_spot_price
    S0 = fetch_spot_price("AAPL")
    return S0

safe("spot_price", test_spot)


# ============================
# OHLC TEST
# ============================

def test_ohlc():
    from app.model.market_data.market_data import fetch_closing_prices
    df = fetch_closing_prices("AAPL", period="2y", interval="1d")
    return len(df)

safe("ohlc", test_ohlc)


# ============================
# YAHOO OPTION CHAIN
# ============================

def test_yahoo():
    from app.model.market_data.market_data import fetch_options_details
    calls, puts, spot, rf, div = fetch_options_details("AAPL")
    return f"calls={len(calls)}, puts={len(puts)}, spot={spot}, rf={rf}, div={div}"

safe("yahoo_options", test_yahoo)


# ============================
# IV SURFACE GENERATION
# ============================

def test_iv_surface():
    from app.model.options.data.iv_surface import fetch_iv_surface
    df = fetch_iv_surface("AAPL")
    return df.head().to_string()

safe("iv_surface", test_iv_surface)


# ============================
# ALPACA OPTIONS
# ============================

def test_alpaca():
    from app.model.options.logic import download_options_alpaca
    df = download_options_alpaca("AAPL")
    return f"rows={len(df)}"

safe("alpaca_options", test_alpaca)


# ============================
# MONTE CARLO PRICE
# ============================

def test_mc():
    from app.model.market_data.market_data import fetch_spot_price
    from app.model.options.mc_engine import price_european_mc
    S0 = fetch_spot_price("AAPL")
    price = price_european_mc(S0, K=S0, T=0.5, sigma=0.3, model="bs", ticker="AAPL", n_paths=2000, n_steps=64)
    return price

safe("mc_price", test_mc)


# ============================
# RESULT EXPORT
# ============================

for k, v in results.items():
    status, msg = v
    print(f"<<RESULT>> {k} | {status} | {msg}")
"@
$tempFile = Join-Path $repoRoot "scripts/test_api_temp.py"
Set-Content -Path $tempFile -Value $py -Encoding UTF8


# Execute Python script
try {
    $execFailed = $false
    Push-Location $repoRoot
    try {
        $output = python $tempFile 2>&1
    } catch {
        $execFailed = $true
        $errorMsg = $_
    } finally {
        Pop-Location
    }

# Remove temporary file
    Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    # Recreate placeholder to satisfy repository checks
    "pass" | Set-Content $tempFile -Encoding UTF8

    if ($execFailed) {
        Write-FAIL "Python execution failed: $errorMsg"
        exit 1
    }


    Write-Host "`n=== API TEST REPORT ===" -ForegroundColor Cyan

$success = $true

    foreach ($line in $output) {
        if ($line -like "<<RESULT>>*") {
            $parts = $line.Replace("<<RESULT>>","").Trim() -split "\|"
            $name = $parts[0].Trim()
            $status = $parts[1].Trim()
            $msg = $parts[2].Trim()

            if ($status -eq "OK") {
                Write-OK "$name : $msg"
            } else {
                Write-FAIL "$name : $msg"
                $success = $false
            }
        }
    }

Write-Host ""
    if ($success) {
        Write-Host "=== ALL TESTS PASSED SUCCESSFULLY ===" -ForegroundColor Green
    } else {
        Write-Host "=== SOME TESTS FAILED (SEE ABOVE) ===" -ForegroundColor Red
    }
} finally {
    if ($startedTranscript) {
        try { Stop-Transcript | Out-Null } catch { }
        Remove-Variable -Name "__test_transcript_active" -Scope Global -ErrorAction SilentlyContinue
    }
}
