Write-Host "=== RUNNING SELECTED TESTS ===" -ForegroundColor Cyan

$repoRoot = $PSScriptRoot
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
	        Write-Warning "Transcript start failed for ${scriptName}: $_"
	    }
	}

# Liste des tests a executer (desormais dans scripts/)
# Auto-decouverte pour eviter les references a des fichiers supprimes.
$scriptsDir = Join-Path $repoRoot "scripts"
$scriptsDirExists = Test-Path -Path $scriptsDir
if (-not $scriptsDirExists) {
    Write-Error "Tests directory not found: $scriptsDir"
    exit 1
}
$env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"

# Prefer a direct pytest entrypoint when available, otherwise fall back to `python -m pytest`
$runnerExe = $null
$runnerArgs = @()
$pytestCmd = Get-Command pytest -ErrorAction SilentlyContinue
if ($pytestCmd) {
    $runnerExe = $pytestCmd.Source
} else {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command py -ErrorAction SilentlyContinue
    }
    if (-not $pythonCmd) {
        Write-Error "Neither 'pytest' nor a Python launcher ('python'/'py') was found on PATH."
        exit 1
    }
    $runnerExe = $pythonCmd.Source
    $runnerArgs = @("-m", "pytest")
}

try {
    if ($pytestCmd) {
        & $runnerExe --version | Out-Null
    } else {
        & $runnerExe @runnerArgs --version | Out-Null
    }
} catch {
    Write-Error "Pytest is not available. Install it with: python -m pip install -U pytest"
    exit 1
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "Pytest is not available in this environment. Install it with: python -m pip install -U pytest"
    exit 1
}
$tests = Get-ChildItem -Path $scriptsDir -Filter "test_*.py" -File |
    Where-Object { $_.Name -ne "test_all.py" } |
    Select-Object -ExpandProperty FullName

$failed = 0

try {
    Push-Location $repoRoot
	    try {
	        foreach ($test in $tests) {
	            $name = Split-Path $test -Leaf
	            Write-Host "`n--- RUNNING $name ---" -ForegroundColor Yellow

	            & $runnerExe @runnerArgs $test -q
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
