# ======================================================
# test_options_panels.ps1
# Test du code des sous-onglets d'Options sans UI
# ======================================================

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

function OK($m) { Write-Host "[OK] $m" -ForegroundColor Green }
function FAIL($m) { Write-Host "[FAIL] $m" -ForegroundColor Red }

Write-Host "=== Testing Options Panel Modules ===" -ForegroundColor Cyan

# Create temporary Python harness
$py = @"
import traceback
import streamlit as st
from types import SimpleNamespace

results = {}

def safe(name, fn):
    try:
        fn()
        results[name] = ("OK", "")
    except Exception as e:
        results[name] = ("FAIL", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# Fake minimal Streamlit context so UI modules import
class FakeState(dict):
    def __getattr__(self, x):
        return self.get(x, None)

st.session_state = FakeState()
st.write = lambda *args, **kwargs: None
st.line_chart = lambda *args, **kwargs: None
st.metric = lambda *args, **kwargs: None
st.selectbox = lambda *args, **kwargs: list(args)[0] if args else None
st.slider = lambda *args, **kwargs: 0
st.columns = lambda x: [SimpleNamespace() for _ in range(x)]
st.markdown = lambda *args, **kwargs: None


# Import panel modules
def test_imports():
    import app.vue.components.options.panels.spreads.tab_straddle
    import app.vue.components.options.panels.spreads.tab_strangle
    import app.vue.components.options.panels.spreads.tab_butterfly
    import app.vue.components.options.panels.spreads.tab_call_spread
    import app.vue.components.options.panels.spreads.tab_condor
    import app.vue.components.options.panels.calendar.tab_calendar
    return "imports OK"

safe("imports", test_imports)


# Test rendering of each panel function
def test_panel_render(modname, funcname):
    mod = __import__(modname, fromlist=[funcname])
    fn = getattr(mod, funcname)
    fn()  # No args expected normally


safe("straddle_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.spreads.tab_straddle",
    "render_tab_straddle"
))

safe("strangle_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.spreads.tab_strangle",
    "render_tab_strangle"
))

safe("butterfly_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.spreads.tab_butterfly",
    "render_tab_butterfly"
))

safe("vertical_spread", lambda: test_panel_render(
    "app.vue.components.options.panels.spreads.tab_call_spread",
    "render_tab_call_spread"
))

safe("condor_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.spreads.tab_condor",
    "render_tab_condor"
))

safe("calendar_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.calendar.tab_calendar",
    "render_tab_calendar"
))


# Export results
for k, v in results.items():
    status, msg = v
    print(f"<<RESULT>> {k} | {status} | {msg}")
"@

$tempFile = Join-Path $PSScriptRoot "test_options_panels_temp.py"
Set-Content $tempFile $py

try {
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

    Write-Host "`n=== OPTIONS PANELS TEST REPORT ===" -ForegroundColor Cyan

    $success = $true
    foreach ($line in $output) {
        if ($line -like "<<RESULT>>*") {
            $parts = $line.Replace("<<RESULT>>","").Trim() -split "\|"
            $name = $parts[0].Trim(); $status = $parts[1].Trim(); $msg = $parts[2].Trim()

            if ($status -eq "OK") { OK "$name" }
            else { FAIL "$name - $msg"; $success = $false }
        }
    }

    Write-Host ""
    if ($success) { Write-Host "=== ALL PANELS LOAD & RENDER OK ===" -ForegroundColor Green }
    else { Write-Host "=== PANEL ERRORS DETECTED ===" -ForegroundColor Red }
} finally {
    if ($startedTranscript) {
        try { Stop-Transcript | Out-Null } catch { }
        Remove-Variable -Name "__test_transcript_active" -Scope Global -ErrorAction SilentlyContinue
    }
}
