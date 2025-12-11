# ============================================================
# run_all_tests.ps1
# Master test runner for PaperTradingApp
# Runs:
#   1) test_API.ps1
#   2) test_full_UI_crawler.ps1
#   3) test_options_context.ps1
#   4) test_options_panels.ps1
# ============================================================

$ErrorActionPreference = "Stop"

function OK($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function FAIL($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red }

$tests = @(
    "test_API.ps1",
    "test_full_UI_crawler.ps1",
    "test_options_context.ps1",
    "test_options_panels.ps1"
)

Write-Host "=== RUNNING FULL TEST SUITE ===" -ForegroundColor Cyan
Write-Host ""

$globalSuccess = $true
$results = @{}

foreach ($test in $tests) {
    Write-Host "----------------------------------------" -ForegroundColor DarkCyan
    Write-Host "Running $test ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor DarkCyan

    if (!(Test-Path $test)) {
        FAIL "$test NOT FOUND"
        $results[$test] = "MISSING"
        $globalSuccess = $false
        continue
    }

    try {
        $output = & ".\$test" 2>&1

        if ($LASTEXITCODE -ne 0) {
            FAIL "$test FAILED (exit code $LASTEXITCODE)"
            $results[$test] = "FAIL"
            $globalSuccess = $false
        }
        else {
            OK "$test completed successfully"
            $results[$test] = "OK"
        }
    }
    catch {
        FAIL "$test crashed: $_"
        $results[$test] = "FAIL"
        $globalSuccess = $false
    }

    Write-Host ""
}

Write-Host ""
Write-Host "============ FINAL TEST REPORT ============" -ForegroundColor Cyan

foreach ($kvp in $results.GetEnumerator()) {
    $name = $kvp.Key
    $status = $kvp.Value

    if ($status -eq "OK") {
        OK "$name"
    }
    elseif ($status -eq "MISSING") {
        FAIL "$name (file missing)"
    }
    else {
        FAIL "$name"
    }
}

Write-Host ""
if ($globalSuccess) {
    Write-Host "=== ALL TESTS PASSED SUCCESSFULLY ===" -ForegroundColor Green
} else {
    Write-Host "=== SOME TESTS FAILED — CHECK REPORT ABOVE ===" -ForegroundColor Red
}
