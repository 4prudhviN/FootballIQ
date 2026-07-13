<#
╔═══════════════════════════════════════════════════════════════════════════════╗
║  FootballIQ — Unified Launch Script (Windows / PowerShell)                  ║
║                                                                             ║
║  This script automates the complete local deployment of the FootballIQ      ║
║  stack on Windows:                                                          ║
║                                                                             ║
║    Step A — Set up Python virtual environment & install backend deps        ║
║    Step B — Start the FastAPI analysis server on http://localhost:8000      ║
║    Step C — Start the React/Vite frontend on http://localhost:5173          ║
║                                                                             ║
║  Prerequisites:                                                             ║
║    - Python 3.10+  (add to PATH during installation)                        ║
║    - Node.js 18+   (includes npm)                                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
#>

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║        FootballIQ — Boot Sequence Initiated          ║" -ForegroundColor Green
Write-Host "  ║   Democratizing Elite Sports Intelligence             ║" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# ── Step A — Python backend environment ──────────────────────────────────────
Write-Host "[INFO]  Step A: Setting up Python backend environment..." -ForegroundColor Cyan

$VenvDir = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO]  Creating Python virtual environment at .venv..." -ForegroundColor Cyan
    python -m venv $VenvDir
    Write-Host "[OK]    Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[OK]    Virtual environment already exists." -ForegroundColor Green
}

# Activate and upgrade pip.
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
& $ActivateScript

Write-Host "[INFO]  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade pip

# Install dependencies.
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
if (Test-Path $RequirementsFile) {
    Write-Host "[INFO]  Installing Python dependencies from requirements.txt..." -ForegroundColor Cyan
    pip install --quiet -r $RequirementsFile
    Write-Host "[OK]    Python dependencies installed." -ForegroundColor Green
} else {
    Write-Host "[ERROR] requirements.txt not found at $RequirementsFile" -ForegroundColor Red
    exit 1
}

# Ensure backend_temp directory exists.
$TempDir = Join-Path $ProjectRoot "backend_temp"
if (-not (Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir | Out-Null
}
Write-Host "[OK]    Backend temp directory ready." -ForegroundColor Green
Write-Host ""

# ── Step B — Start FastAPI backend server ────────────────────────────────────
Write-Host "[INFO]  Step B: Starting FastAPI backend on http://localhost:8000..." -ForegroundColor Cyan

$BackendJob = Start-Job -ScriptBlock {
    param($Root)
    Set-Location $Root
    $Activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
    & $Activate
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload --log-level info
} -ArgumentList $ProjectRoot

Start-Sleep -Seconds 3
Write-Host "[OK]    FastAPI server started (Job ID: $($BackendJob.Id))" -ForegroundColor Green

# Quick health-check.
try {
    $Response = Invoke-WebRequest -Uri "http://localhost:8000/" -UseBasicParsing -TimeoutSec 5
    Write-Host "[OK]    Backend health-check passed." -ForegroundColor Green
} catch {
    Write-Host "[WARN]  Backend health-check failed — it may still be starting." -ForegroundColor Yellow
}

Write-Host ""

# ── Step C — Frontend dev server ─────────────────────────────────────────────
Write-Host "[INFO]  Step C: Installing frontend deps and starting Vite dev server..." -ForegroundColor Cyan

$NodeModules = Join-Path $ProjectRoot "node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Host "[INFO]  Installing npm packages..." -ForegroundColor Cyan
    npm install
    Write-Host "[OK]    npm packages installed." -ForegroundColor Green
} else {
    Write-Host "[OK]    node_modules already exists (run 'npm install' if needed)." -ForegroundColor Green
}

# Start Vite in a background job.
$FrontendJob = Start-Job -ScriptBlock {
    param($Root)
    Set-Location $Root
    npm run dev
} -ArgumentList $ProjectRoot

Start-Sleep -Seconds 4
Write-Host "[OK]    Vite dev server started (Job ID: $($FrontendJob.Id))" -ForegroundColor Green

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║   🚀 FootballIQ Fully Online                        ║" -ForegroundColor Green
Write-Host "  ║                                                    ║" -ForegroundColor Green
Write-Host "  ║   Frontend   →  http://localhost:5173               ║" -ForegroundColor Green
Write-Host "  ║   Backend    →  http://localhost:8000               ║" -ForegroundColor Green
Write-Host "  ║   API Docs   →  http://localhost:8000/docs          ║" -ForegroundColor Green
Write-Host "  ║                                                    ║" -ForegroundColor Green
Write-Host "  ║   Press Ctrl+C to stop the terminal session.        ║" -ForegroundColor Green
Write-Host "  ║   To stop background jobs manually:                 ║" -ForegroundColor Green
Write-Host "  ║     Stop-Job -Id $($BackendJob.Id)                              ║" -ForegroundColor Green
Write-Host "  ║     Stop-Job -Id $($FrontendJob.Id)                             ║" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Keep the script alive until the user presses Ctrl+C.
Write-Host "Press Ctrl+C to stop all services..." -ForegroundColor Gray
while ($true) {
    Start-Sleep -Seconds 10
}