Write-Host "=== RUNNING SELECTED TESTS ===" -ForegroundColor Cyan

# Liste des tests à exécuter
$tests = @(
    "tests/test_heston_separation.py",
    "tests/test_options_model_integrity.py",
    "tests/test_options_pricing_core.py",
    "tests/test_portfolio_valuation.py",
    "tests/test_yieldcurve_builder.py"
)

$failed = 0

foreach ($test in $tests) {
    Write-Host "`n--- RUNNING $test ---" -ForegroundColor Yellow

    pytest $test -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAILED] $test" -ForegroundColor Red
        $failed += 1
    } else {
        Write-Host "[OK] $test" -ForegroundColor Green
    }
}

Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
if ($failed -gt 0) {
    Write-Host "$failed test(s) failed." -ForegroundColor Red
    exit 1
} else {
    Write-Host "All tests passed successfully." -ForegroundColor Green
    exit 0
}
