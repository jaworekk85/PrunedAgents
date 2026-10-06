param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Torch = "auto",

    [switch]$CacheModel,

    [switch]$SkipSmoke,

    [string]$ModelConfig = "configs\experiments\hf_generation_test.yaml"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-External {
    param(
        [string]$Exe,
        [string[]]$Arguments
    )
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        $CommandText = "$Exe $($Arguments -join ' ')"
        throw "Command failed with exit code ${LASTEXITCODE}: $CommandText"
    }
}

function Test-NvidiaGpu {
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidia) {
        return $false
    }
    try {
        & nvidia-smi | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$UseCuda = $false
if ($Torch -eq "cuda") {
    $UseCuda = $true
} elseif ($Torch -eq "auto") {
    $UseCuda = Test-NvidiaGpu
}

Write-Step "Creating local virtual environment"
if (-not (Test-Path ".venv")) {
    Invoke-External "python" @("-m", "venv", ".venv")
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Could not find venv Python at $Python"
}

Write-Step "Upgrading packaging tools"
Invoke-External $Python @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

if ($UseCuda) {
    Write-Step "Installing CUDA PyTorch (cu124)"
    Invoke-External $Python @("-m", "pip", "install", "-r", "requirements\torch-cu124.txt")
} else {
    Write-Step "Using CPU PyTorch from project dependencies"
}

Write-Step "Installing project dependencies"
if ($UseCuda) {
    Invoke-External $Python @("-m", "pip", "install", "--no-build-isolation", "-e", ".[dev,gpu]")
} else {
    Invoke-External $Python @("-m", "pip", "install", "--no-build-isolation", "-r", "requirements\dev.txt")
}

Write-Step "Checking dependency consistency"
Invoke-External $Python @("-m", "pip", "check")

Write-Step "Verifying environment"
if ($UseCuda) {
    Invoke-External $Python @("scripts\verify_environment.py", "--expect-cuda")
} else {
    Invoke-External $Python @("scripts\verify_environment.py")
}

Write-Step "Running unit tests"
Invoke-External $Python @("-m", "unittest", "discover", "tests")

if (-not $SkipSmoke) {
    Write-Step "Running local smoke pipeline"
    Invoke-External $Python @("-m", "role_pruning.cli", "smoke", "--config", "configs\experiments\smoke.yaml")
}

if ($CacheModel) {
    Write-Step "Caching/testing Hugging Face model"
    Invoke-External $Python @("-m", "role_pruning.cli", "hf-generation-test", "--config", $ModelConfig)
}

Write-Step "Setup complete"
Write-Host "Use this Python:" -ForegroundColor Green
Write-Host $Python
