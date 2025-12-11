Write-Host "=== RUNNING SELECTED TESTS ===" -ForegroundColor Cyan

$repoRoot = Split-Path -Parent $PSScriptRoot
# Liste des tests a executer (desormais dans scripts/)
$tests = @(
    "test_heston_separation.py",
    "test_options_model_integrity.py",
    "test_options_pricing_core.py",
    "test_portfolio_valuation.py",
    "test_yieldcurve_builder.py"
) | ForEach-Object { Join-Path $repoRoot "scripts/$_" }

$failed = 0

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
