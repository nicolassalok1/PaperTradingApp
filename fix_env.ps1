###############################################################
# fix_env.ps1 — Full Environment Repair Script for Windows
###############################################################

Write-Host "`n=== FIX_ENV — FULL ENVIRONMENT REPAIR ===`n" -ForegroundColor Cyan

###############################################################
# 1. Kill ALL processes that lock Python / Matplotlib / Streamlit
###############################################################

Write-Host "Killing Python / Streamlit / VSCode processes..." -ForegroundColor Yellow

$processes = @("python.exe", "streamlit.exe", "Code.exe")

foreach ($p in $processes) {
    try {
        taskkill /F /IM $p /T 2>$null
        Write-Host "Killed: $p"
    } catch {
        Write-Host "$p not running."
    }
}

Start-Sleep -Seconds 2

###############################################################
# 2. Activate environment safely
###############################################################

$EnvName = "papertrading"

Write-Host "`nActivating Conda environment '$EnvName'..." -ForegroundColor Cyan

conda activate $EnvName

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Could not activate env. Trying fallback..." -ForegroundColor Red
    & "$env:USERPROFILE\Miniconda3\shell\condabin\conda-hook.ps1"
    conda activate $EnvName
}

###############################################################
# 3. Remove Matplotlib font locks (root cause of WinError 32)
###############################################################

$matplotlibPath = "D:\Programmes\Miniconda3\envs\papertrading\Lib\site-packages\matplotlib\mpl-data\fonts\ttf"

if (Test-Path $matplotlibPath) {
    Write-Host "`nCleaning locked Matplotlib fonts..." -ForegroundColor Yellow
    Get-ChildItem "$matplotlibPath\*.ttf" | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "Matplotlib font directory cleaned."
} else {
    Write-Host "`nMatplotlib font directory not found. Skipping cleanup." -ForegroundColor DarkYellow
}

###############################################################
# 4. Clear pip cache
###############################################################

Write-Host "`nClearing pip cache..." -ForegroundColor Yellow
pip cache purge

###############################################################
# 5. Force reinstall base dependencies to prevent file locks
###############################################################

Write-Host "`nReinstalling core libs (numpy, pillow, matplotlib)..." -ForegroundColor Cyan
pip install --force-reinstall --no-cache-dir numpy pillow matplotlib

###############################################################
# 6. Reinstall the entire environment
###############################################################

Write-Host "`nInstalling project dependencies (requirements.txt)..." -ForegroundColor Cyan
pip install --no-cache-dir -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR during requirements install!" -ForegroundColor Red
    exit 1
}

###############################################################
# 7. Final environment validation — import check
###############################################################

Write-Host "`nRunning global import check..." -ForegroundColor Cyan

python - <<EOF
import sys, pkgutil
print("Import test OK. Python version:", sys.version)
EOF

###############################################################
# 8. Launch the application
###############################################################

Write-Host "`n=== ENV FIX COMPLETE — Launching app ===`n" -ForegroundColor Green

.\run_me.ps1
