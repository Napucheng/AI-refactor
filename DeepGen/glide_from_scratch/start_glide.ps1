param(
    [ValidateSet("setup", "train", "sample", "download-tokenizer")]
    [string]$Mode = "train",
    [string]$VenvDir = ".venv",
    [string]$PythonExe = "python",
    [string]$RepoRoot = "",
    [string]$DataRoot = "",
    [string]$Metadata = "",
    [string]$TokenizerPath = "checkpoints\hf\clip-vit-base-patch32",
    [string]$OutputDir = "checkpoints\glide_tiny",
    [string]$Checkpoint = "checkpoints\glide_tiny\model_latest.pt",
    [string]$Prompt = "a cozy cabin in snowy mountains at dusk",
    [int]$ImageSize = 64,
    [int]$BatchSize = 8,
    [int]$Epochs = 50,
    [int]$Timesteps = 1000,
    [int]$NumSamples = 4,
    [double]$GuidanceScale = 5.0,
    [int]$SampleSteps = 100
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[GLIDE] $Message" -ForegroundColor Cyan
}

function Resolve-RepoRoot {
    if ($RepoRoot -and $RepoRoot.Trim().Length -gt 0) {
        return (Resolve-Path $RepoRoot).Path
    }
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-VenvPython {
    param([string]$Root, [string]$EnvName, [string]$FallbackPython)
    $venvPython = Join-Path $Root "$EnvName\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return $FallbackPython
}

function Ensure-Venv {
    param([string]$Root, [string]$EnvName, [string]$FallbackPython)
    $venvPython = Join-Path $Root "$EnvName\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Step "Creating virtual environment at $EnvName"
        & $FallbackPython -m venv (Join-Path $Root $EnvName)
    }
    return $venvPython
}

function Install-Requirements {
    param([string]$PythonCmd, [string]$Root)
    Write-Step "Installing requirements"
    & $PythonCmd -m pip install --upgrade pip
    & $PythonCmd -m pip install -r (Join-Path $Root "glide_from_scratch\requirements.txt")
}

function Ensure-Tokenizer {
    param([string]$PythonCmd, [string]$Root, [string]$LocalTokenizerPath)
    $fullTokenizerPath = Join-Path $Root $LocalTokenizerPath
    if (-not (Test-Path $fullTokenizerPath)) {
        Write-Step "Downloading CLIP tokenizer to $LocalTokenizerPath"
        & $PythonCmd -m glide_from_scratch.download_hf_assets `
            --repo-id openai/clip-vit-base-patch32 `
            --local-dir $fullTokenizerPath
    }
    else {
        Write-Step "Tokenizer already exists at $LocalTokenizerPath"
    }
}

$Root = Resolve-RepoRoot
Set-Location $Root

if ($Mode -eq "setup") {
    $PythonCmd = Ensure-Venv -Root $Root -EnvName $VenvDir -FallbackPython $PythonExe
    Install-Requirements -PythonCmd $PythonCmd -Root $Root
    Ensure-Tokenizer -PythonCmd $PythonCmd -Root $Root -LocalTokenizerPath $TokenizerPath
    Write-Step "Setup complete"
    exit 0
}

$PythonCmd = Ensure-Venv -Root $Root -EnvName $VenvDir -FallbackPython $PythonExe
Install-Requirements -PythonCmd $PythonCmd -Root $Root

if ($Mode -eq "download-tokenizer") {
    Ensure-Tokenizer -PythonCmd $PythonCmd -Root $Root -LocalTokenizerPath $TokenizerPath
    Write-Step "Tokenizer download complete"
    exit 0
}

Ensure-Tokenizer -PythonCmd $PythonCmd -Root $Root -LocalTokenizerPath $TokenizerPath

if ($Mode -eq "train") {
    if (-not $DataRoot) {
        throw "Please pass -DataRoot for training."
    }
    if (-not $Metadata) {
        throw "Please pass -Metadata for training."
    }
    Write-Step "Starting training"
    & $PythonCmd -m glide_from_scratch.train `
        --data-root $DataRoot `
        --metadata $Metadata `
        --tokenizer-path $TokenizerPath `
        --output-dir $OutputDir `
        --image-size $ImageSize `
        --batch-size $BatchSize `
        --epochs $Epochs `
        --timesteps $Timesteps `
        --sample-prompts "a golden retriever wearing sunglasses" "a futuristic city at sunrise"
    exit $LASTEXITCODE
}

if ($Mode -eq "sample") {
    if (-not (Test-Path $Checkpoint)) {
        throw "Checkpoint not found: $Checkpoint"
    }
    Write-Step "Starting sampling"
    & $PythonCmd -m glide_from_scratch.sample `
        --checkpoint $Checkpoint `
        --tokenizer-path $TokenizerPath `
        --prompt $Prompt `
        --num-samples $NumSamples `
        --guidance-scale $GuidanceScale `
        --sampler ddim `
        --steps $SampleSteps `
        --output-dir "outputs\glide_samples"
    exit $LASTEXITCODE
}
