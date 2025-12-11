Write-Host "=== Running full test suite (scripts/test_all.py) ==="

$repoRoot = Split-Path -Parent $PSScriptRoot
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
        Write-Warning "Transcript start failed for $scriptName: $_"
    }
}

$testPath = Join-Path $repoRoot "scripts/test_all.py"

try {
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
} finally {
    if ($startedTranscript) {
        try { Stop-Transcript | Out-Null } catch { }
        Remove-Variable -Name "__test_transcript_active" -Scope Global -ErrorAction SilentlyContinue
    }
}
