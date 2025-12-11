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

$repoRoot = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$scriptName = [IO.Path]::GetFileNameWithoutExtension($PSCommandPath)
$logFile = Join-Path $logsDir "$scriptName.log"

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

function OK($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function FAIL($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red }

$tests = @(
    "test_API.ps1",
    "test_full_UI_crawler.ps1",
    "test_options_context.ps1",
    "test_options_panels.ps1"
) | ForEach-Object { Join-Path $PSScriptRoot $_ }

try {
    Write-Host "=== RUNNING FULL TEST SUITE ===" -ForegroundColor Cyan
    Write-Host ""

    $globalSuccess = $true
    $results = @{}

    foreach ($test in $tests) {
        $name = Split-Path $test -Leaf
        Write-Host "----------------------------------------" -ForegroundColor DarkCyan
        Write-Host "Running $name ..." -ForegroundColor Cyan
        Write-Host "----------------------------------------" -ForegroundColor DarkCyan

        if (!(Test-Path $test)) {
            FAIL "$name NOT FOUND"
            $results[$name] = "MISSING"
            $globalSuccess = $false
            continue
        }

        try {
            $output = & $test 2>&1

            if ($LASTEXITCODE -ne 0) {
                FAIL "$name FAILED (exit code $LASTEXITCODE)"
                $results[$name] = "FAIL"
                $globalSuccess = $false
            }
            else {
                OK "$name completed successfully"
                $results[$name] = "OK"
            }
        }
        catch {
            FAIL "$name crashed: $_"
            $results[$name] = "FAIL"
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
        Write-Host "=== SOME TESTS FAILED - CHECK REPORT ABOVE ===" -ForegroundColor Red
    }
} finally {
    if ($startedTranscript) {
        try { Stop-Transcript | Out-Null } catch { }
        Remove-Variable -Name "__test_transcript_active" -Scope Global -ErrorAction SilentlyContinue
    }
}
