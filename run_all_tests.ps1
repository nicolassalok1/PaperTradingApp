#!/usr/bin/env pwsh
# Run the full test suite under ./tests and stream logs to the terminal (PowerShell).

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $repoRoot

Write-Host "=== Running tests via unittest discover (tests/)..."
python -m unittest discover -s tests -p "test*.py" -t . @args | Tee-Object -FilePath "tests/testsGPT/test_run.log"
