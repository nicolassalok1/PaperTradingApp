Write-Host "=== Running full test suite (test_all.py) ==="

$envName = "papertrading"
try {
    conda activate $envName
} catch {
    Write-Warning "Conda activate failed; ensure the 'papertrading' env is available."
}

try {
    python test_all.py
    Write-Host "TEST SUITE COMPLETED"
} catch {
    Write-Error "Tests failed: $_"
    exit 1
}
