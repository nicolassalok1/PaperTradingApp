# Master runner: executes every Python test script under scripts/ once.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSCommandPath
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
        Write-Warning "Transcript start failed for $($scriptName): $_"
    }
}

$scriptsDir = Join-Path $repoRoot "scripts"
$env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"

$pyTests = @()
if (Test-Path $scriptsDir) {
    $pyTests = Get-ChildItem -Path $scriptsDir -Filter "test_*.py" -File -Recurse |
        Sort-Object FullName -Unique
}

try {
    if (-not $pyTests) {
        Write-Warning "No Python tests found in scripts/."
        exit 0
    }

    $script:results = @()

    function Invoke-PythonTest {
        param([string]$Path)

        $label = Split-Path $Path -Leaf
        Write-Host "`n--- Running $label ---" -ForegroundColor Cyan

        $exitCode = 0
        $output = ""

        Push-Location $repoRoot
        try {
            $output = & python $Path 2>&1
            $exitCode = $LASTEXITCODE
            if ($null -eq $exitCode) { $exitCode = 0 }
        } catch {
            $output = $_ | Out-String
            $exitCode = 1
        } finally {
            Pop-Location
        }

        if ($output) {
            Write-Host $output
        }

        $status = if ($exitCode -eq 0) { "OK" } else { "FAIL" }
        $script:results += [pscustomobject]@{
            Name     = $label
            Path     = $Path
            Status   = $status
            ExitCode = $exitCode
            Output   = $output
        }

        if ($exitCode -eq 0) {
            Write-Host "[OK] $label" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] $label (exit $exitCode)" -ForegroundColor Red
        }
    }

    foreach ($py in $pyTests) {
        Invoke-PythonTest -Path $py.FullName
    }

    Write-Host "`n=== TEST SUMMARY ===" -ForegroundColor Cyan
    $failures = @()
    foreach ($res in $results) {
        if ($res.Status -eq "OK") {
            Write-Host "[OK] $($res.Name)" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] $($res.Name)" -ForegroundColor Red
            $failures += $res
        }
    }

    if ($failures.Count -gt 0) {
        Write-Host "`nFailures: $($failures.Count)" -ForegroundColor Red
        exit 1
    }

    Write-Host "`nAll tests passed." -ForegroundColor Green
    exit 0
} finally {
    if ($startedTranscript) {
        try { Stop-Transcript | Out-Null } catch { }
        Remove-Variable -Name "__test_transcript_active" -Scope Global -ErrorAction SilentlyContinue
    }
}
