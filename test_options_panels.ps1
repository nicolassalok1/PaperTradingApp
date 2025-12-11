# ======================================================
# test_options_panels.ps1
# Test du code des sous-onglets d'Options sans UI
# ======================================================

$ErrorActionPreference = "Stop"

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
    import app.vue.components.options.panels.tab_straddle
    import app.vue.components.options.panels.tab_strangle
    import app.vue.components.options.panels.tab_butterfly
    import app.vue.components.options.panels.tab_vertical_spread
    import app.vue.components.options.panels.tab_condor
    import app.vue.components.options.panels.tab_calendar
    return "imports OK"

safe("imports", test_imports)


# Test rendering of each panel function
def test_panel_render(modname, funcname):
    mod = __import__(modname, fromlist=[funcname])
    fn = getattr(mod, funcname)
    fn()  # No args expected normally


safe("straddle_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.tab_straddle",
    "render_straddle_panel"
))

safe("strangle_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.tab_strangle",
    "render_strangle_panel"
))

safe("butterfly_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.tab_butterfly",
    "render_butterfly_panel"
))

safe("vertical_spread", lambda: test_panel_render(
    "app.vue.components.options.panels.tab_vertical_spread",
    "render_vertical_spread_panel"
))

safe("condor_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.tab_condor",
    "render_condor_panel"
))

safe("calendar_panel", lambda: test_panel_render(
    "app.vue.components.options.panels.tab_calendar",
    "render_calendar_panel"
))


# Export results
for k, v in results.items():
    status, msg = v
    print(f"<<RESULT>> {k} | {status} | {msg}")
"@

$tempFile = "test_options_panels_temp.py"
Set-Content $tempFile $py


try {
    $output = python $tempFile 2>&1
} catch {
    FAIL "Python execution failed: $_"
    exit 1
}

Remove-Item $tempFile -Force -ErrorAction SilentlyContinue

Write-Host "`n=== OPTIONS PANELS TEST REPORT ===" -ForegroundColor Cyan

$success = $true
foreach ($line in $output) {
    if ($line -like "<<RESULT>>*") {
        $parts = $line.Replace("<<RESULT>>","").Trim() -split "\|"
        $name = $parts[0].Trim(); $status = $parts[1].Trim(); $msg = $parts[2].Trim()

        if ($status -eq "OK") { OK "$name" }
        else { FAIL "$name — $msg"; $success = $false }
    }
}

Write-Host ""
if ($success) { Write-Host "=== ALL PANELS LOAD & RENDER OK ===" -ForegroundColor Green }
else { Write-Host "=== PANEL ERRORS DETECTED ===" -ForegroundColor Red }
