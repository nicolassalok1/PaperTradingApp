# ===============================
# run_me.ps1 — RUN STREAMLIT APP
# ===============================

Write-Host "`n=== Starting PaperTradingApp ===`n" -ForegroundColor Cyan

# 1) Activate environment
$envName = "papertrading"
conda activate $envName

# 1bis) Charger .env dans l'environnement courant (pour Streamlit/Alpaca)
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*#") { return }
        if ($_ -match "^\s*$") { return }
        $parts = $_ -split "=", 2
        if ($parts.Length -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1]
            Set-Item -Path "env:$name" -Value $value
        }
    }
}

# 2) Optional: preflight import check
Write-Host "Running MVC import check..." -ForegroundColor Cyan
@'
import sys, pkgutil, importlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")

print("Checking imports...")
# Simple test: try to import app.vue.main_app
try:
    import app.vue.main_app
    print("MVC OK.")
except Exception as e:
    print("MVC Import Error:")
    print(e)
    sys.exit(1)
'@ | python -

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

# 3) Launch Streamlit
Write-Host "`nLaunching Streamlit..." -ForegroundColor Green
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"
streamlit run app/vue/main_app.py
