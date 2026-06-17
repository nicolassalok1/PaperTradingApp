<#
.SYNOPSIS
  Phase 0 capture for the runtime console-error hunt (see CONSOLE_ERRORS.md).
  Launches the Streamlit app headless in the conda `papertrading` env with maximal
  warning verbosity and writes the full terminal output to logs/console_runtime.log.

.NOTES
  Diagnostic harness. Stop with Ctrl+C; the Streamlit server runs until killed.

  Buffering note: `python -u` (unbuffered) + `conda run --no-capture-output` are both
  required, otherwise stdout/stderr sit in a block buffer and never reach the log file
  while the long-running server is alive. Do NOT pipe through Tee-Object: with
  --no-capture-output the child bypasses the PowerShell pipe and Tee captures nothing.

  `PYTHONWARNINGS=default` surfaces the DeprecationWarning/FutureWarning that are silent
  by default; it is independent of `.streamlit/config.toml` [logger] level (left as-is).
#>

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$env:PYTHONWARNINGS = "default"
$env:PYTHONUNBUFFERED = "1"

conda run --no-capture-output -n papertrading python -u -m streamlit run streamlit_app.py `
  --server.headless true `
  --server.port 8501 `
  --server.runOnSave false `
  *> "logs\console_runtime.log"
