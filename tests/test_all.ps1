Write-Host "=== Running full test suite (scripts/test_all.py) ==="

$repoRoot = Split-Path -Parent $PSScriptRoot
$testPath = Join-Path $repoRoot "scripts/test_all.py"

$envName = "papertrading"
try {
    conda activate $envName
} catch {
    Write-Warning "Conda activate failed; ensure the 'papertrading' env is available."
}

Push-Location $repoRoot
try {
    python $testPath
    Write-Host "TEST SUITE COMPLETED"
} catch {
    Write-Error "Tests failed: $_"
    exit 1
} finally {
    Pop-Location
}
