Write-Host "=== RUNNING SELECTED TESTS ===" -ForegroundColor Cyan

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
        Write-Warning "Transcript start failed for $scriptName: $_"
    }
}

# Liste des tests a executer (desormais dans scripts/)
$tests = @(
    "test_options_model_integrity.py",
    "test_options_pricing_core.py",
    "test_portfolio_valuation.py",
    "test_yieldcurve_builder.py"
) | ForEach-Object { Join-Path $repoRoot "scripts/$_" }

$failed = 0

try {
    Push-Location $repoRoot
    try {
        foreach ($test in $tests) {
            $name = Split-Path $test -Leaf
            Write-Host "`n--- RUNNING $name ---" -ForegroundColor Yellow

            pytest $test -q
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[FAILED] $name" -ForegroundColor Red
                $failed += 1
            } else {
                Write-Host "[OK] $name" -ForegroundColor Green
            }
        }
    } finally {
        Pop-Location
    }

    Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
    if ($failed -gt 0) {
        Write-Host "$failed test(s) failed." -ForegroundColor Red
        exit 1
    } else {
        Write-Host "All tests passed successfully." -ForegroundColor Green
        exit 0
    }
} finally {
    if ($startedTranscript) {
        try { Stop-Transcript | Out-Null } catch { }
        Remove-Variable -Name "__test_transcript_active" -Scope Global -ErrorAction SilentlyContinue
    }
}
