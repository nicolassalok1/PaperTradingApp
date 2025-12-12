# ================================================
# test_options_context.ps1
# Test complet du build context Options
# ================================================

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
function OK($m) { Write-Host "[OK] $m" -ForegroundColor Green }
function FAIL($m) { Write-Host "[FAIL] $m" -ForegroundColor Red }

Write-Host "=== Testing Options Context Builder ===" -ForegroundColor Cyan

# -----------------------------------------
# 1) Generation du script Python temporaire
# -----------------------------------------

$py = @"
import traceback
import json

results = {}


def safe(name, fn):
    try:
        out = fn()
        results[name] = ("OK", out)
    except Exception as e:
        results[name] = ("FAIL", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# -----------------------------------------
# Importer les modules necessaires
# -----------------------------------------
def test_imports():
    import app.model.options.context as ctx
    import app.model.market_data.market_data as md
    import pandas as pd
    return "imports ok"


safe("imports", test_imports)


# -----------------------------------------
# TEST 1 : Construction basique du contexte
# -----------------------------------------
def test_basic_context():
    from app.model.options.context import build_option_context
    ctx = build_option_context("AAPL")
    assert "S0" in ctx, "missing S0"
    assert "ticker" in ctx, "missing ticker"
    assert "close_series" in ctx, "missing close_series"
    assert "_k" in ctx, "missing key builder"
    return {
        "S0": ctx["S0"],
        "ticker": ctx["ticker"],
        "close_size": len(ctx["close_series"]),
    }


safe("basic_context", test_basic_context)


# -----------------------------------------
# TEST 2 : Test _k (key builder)
# -----------------------------------------
def test_key_builder():
    from app.model.options.context import build_option_context
    ctx = build_option_context("AAPL")
    k1 = ctx["_k"]("spread_panel")
    k2 = ctx["_k"]("spread_panel")
    assert k1 == k2, "_k must produce stable keys"
    return k1


safe("key_builder", test_key_builder)


# -----------------------------------------
# TEST 3 : Test override S0 via state mapping
# -----------------------------------------
def test_s0_override_state():
    from app.model.options.context import get_option_context_from_state

    state = {
        "ticker": "AAPL",
        "common_spot_value": 123.45,
    }
    ctx = get_option_context_from_state(state)
    assert abs(float(ctx["S0"]) - 123.45) < 1e-4, "Override S0 not applied"
    return ctx["S0"]


safe("s0_override_state", test_s0_override_state)


# -----------------------------------------
# TEST 4 : Test fallback last close
# -----------------------------------------
def test_s0_from_last_close():
    from app.model.options.context import build_option_context

    ctx = build_option_context("AAPL")
    series = ctx["close_series"]
    last = float(series.iloc[-1])
    assert abs(ctx["S0"] - last) < 1e-8, "S0 should equal last close when no override"
    return {"S0": ctx["S0"], "last_close": last}


safe("s0_last_close", test_s0_from_last_close)


# -----------------------------------------
# TEST 5 : Test fallback spot when no close_series
# -----------------------------------------
def test_s0_from_spot_fallback():
    # Monkeypatch : on simule un close_series vide
    from app.model.options.context import build_option_context
    from app.model.options.context import load_close_series_for_ticker

    def fake_load(ticker):
        import pandas as pd

        return pd.Series([], dtype=float)

    import app.model.options.context as ctx_mod

    ctx_mod.load_close_series_for_ticker = fake_load

    ctx = build_option_context("AAPL")
    assert ctx["S0"] is not None, "Spot fallback failed"
    return ctx["S0"]


safe("s0_spot_fallback", test_s0_from_spot_fallback)


# -----------------------------------------
# FIN - EXPORT DES RESULTATS
# -----------------------------------------
for name, (status, msg) in results.items():
    print(f"<<RESULT>> {name} | {status} | {msg}")
"@

$tempFile = Join-Path $PSScriptRoot "test_options_context_temp.py"
Set-Content -Path $tempFile -Value $py -Encoding UTF8

try {

# -----------------------------------------
# 2) Execution du script python
# -----------------------------------------
$execFailed = $false
Push-Location $repoRoot
try {
    $output = python $tempFile 2>&1
} catch {
    FAIL "Python execution failed: $_"
    $execFailed = $true
} finally {
    Pop-Location
}

Remove-Item $tempFile -Force -ErrorAction SilentlyContinue

if ($execFailed) {
    exit 1
}

Write-Host "`n=== OPTIONS CONTEXT TEST REPORT ===" -ForegroundColor Cyan

# -----------------------------------------
# 3) Analyse des resultats
# -----------------------------------------
$success = $true

foreach ($line in $output) {
    if ($line -like "<<RESULT>>*") {
        $parts = $line.Replace("<<RESULT>>","").Trim() -split "\|"
        $name = $parts[0].Trim()
        $status = $parts[1].Trim()
        $msg = $parts[2].Trim()

        if ($status -eq "OK") {
            OK "$name : $msg"
        } else {
            FAIL "$name : $msg"
            $success = $false
        }
    }
}

Write-Host ""
if ($success) {
    Write-Host "=== ALL OPTIONS CONTEXT TESTS PASSED ===" -ForegroundColor Green
} else {
    Write-Host "=== SOME OPTIONS CONTEXT TESTS FAILED - SEE ABOVE ===" -ForegroundColor Red
}
# ================================================

} finally {
    if ($startedTranscript) {
        try { Stop-Transcript | Out-Null } catch { }
        Remove-Variable -Name "__test_transcript_active" -Scope Global -ErrorAction SilentlyContinue
    }
}
