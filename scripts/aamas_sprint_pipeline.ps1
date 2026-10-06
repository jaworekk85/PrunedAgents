param(
    [string]$Config = "configs\experiments\aamas_sprint_gpu.yaml",
    [switch]$SkipActivations,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

function Invoke-Step {
    param([string]$Name, [string[]]$CommandArgs)
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

if (-not $SkipActivations) {
    Invoke-Step "Collect HF activations" @("-m", "role_pruning.cli", "hf-collect-activations-tiny", "--config", $Config)
}

Invoke-Step "Build masks" @("-m", "role_pruning.cli", "build-masks", "--config", $Config)

if (-not $SkipEval) {
    Invoke-Step "Evaluate masked cross-role matrix" @("-m", "role_pruning.cli", "hf-eval-masked-cross-role-tiny", "--config", $Config)
}
