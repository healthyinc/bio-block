# Create an isolated Python environment for the Bio-Block Python backend (Windows).
# Prefers `uv` when available; falls back to stdlib `venv`.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

if (Test-Path $VenvDir) {
    Write-Host "Virtual environment already exists at $Root\$VenvDir"
} else {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host "Creating virtual environment with uv..."
        uv venv $VenvDir
    } else {
        Write-Host "Creating virtual environment with $PythonBin -m venv..."
        & $PythonBin -m venv $VenvDir
    }
}

$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
. $Activate

python -m pip install --upgrade pip
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv pip install -r requirements.txt
} else {
    pip install -r requirements.txt
}

try {
    python -m spacy download en_core_web_sm
} catch {
    Write-Host "Warning: could not download spaCy model en_core_web_sm"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with:  .\$VenvDir\Scripts\Activate.ps1"
Write-Host "Run with:       uvicorn main:app --reload --port 3002"
