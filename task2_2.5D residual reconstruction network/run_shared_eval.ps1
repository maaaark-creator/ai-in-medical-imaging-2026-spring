$ErrorActionPreference = "Stop"

param(
    [switch]$WaitForNoPython,
    [switch]$ForceCPU
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$pythonExe = "D:\conda_data\envs\AI_in_MI\python.exe"
$evaluatePy = Join-Path $scriptDir "evaluate.py"
$outputDir = Join-Path $repoRoot "outputs\task2\exp_25d_resnet_ctx3_nonzero_shared_train"
$modelPath = Join-Path $outputDir "best_resnet25d.pth"

if ($WaitForNoPython) {
    while ((Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq "python" }).Count -gt 0) {
        Write-Host "Another python process is still running. Waiting 60 seconds..."
        Start-Sleep -Seconds 60
    }
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}
if (-not (Test-Path -LiteralPath $evaluatePy)) {
    throw "Evaluation script not found: $evaluatePy"
}
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "Model checkpoint not found: $modelPath"
}

if ($ForceCPU) {
    $env:CUDA_VISIBLE_DEVICES = ""
}

& $pythonExe $evaluatePy `
    --normalization shared `
    --model-path $modelPath `
    --output-dir $outputDir `
    --batch-size 4 `
    --num-workers 0
