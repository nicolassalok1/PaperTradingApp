$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

python scripts/run_ui_crawler_options.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "run_ui_crawler_options.py a retourné $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "run_ui_crawler_options.py terminé avec succès." -ForegroundColor Green
