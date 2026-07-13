#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FootballIQ — Unified Launch Script (Linux / macOS)
# ═══════════════════════════════════════════════════════════════════════════════
# This script automates the complete local deployment of the FootballIQ stack:
#
#   Step A — Set up Python virtual environment & install backend dependencies
#   Step B — Start the FastAPI analysis server on http://localhost:8000
#   Step C — Start the React/Vite frontend on http://localhost:5173
#
# Prerequisites:
#   - Python 3.10+
#   - Node.js 18+
#   - npm 9+
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

# ── Utility helpers ──────────────────────────────────────────────────────────
info()     { echo -e "${CYAN}[INFO]${NC}  $1"; }
success()  { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()     { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()      { echo -e "${RED}[ERROR]${NC} $1"; }

cleanup() {
    info "Shutting down FootballIQ..."
    if [ -n "${BACKEND_PID:-}" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    info "Goodbye."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Step 0 — Locate project root ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔═══════════════════════════════════════════════════╗"
echo "  ║         FootballIQ — Boot Sequence Initiated      ║"
echo "  ║    Democratizing Elite Sports Intelligence         ║"
echo "  ╚═══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# ── Step A — Python backend environment ──────────────────────────────────────
info "${BOLD}Step A:${NC} Setting up Python backend environment..."

# Create a virtual environment if it doesn't exist.
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment at .venv..."
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created."
else
    success "Virtual environment already exists."
fi

# Activate it.
source "$VENV_DIR/bin/activate"

# Upgrade pip inside the virtual environment.
info "Upgrading pip..."
pip install --quiet --upgrade pip

# Install backend dependencies from requirements.txt.
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    info "Installing Python dependencies from requirements.txt..."
    pip install --quiet -r "$PROJECT_ROOT/requirements.txt"
    success "Python dependencies installed."
else
    err "requirements.txt not found at $PROJECT_ROOT/requirements.txt"
    exit 1
fi

# Create the backend_temp directory (used by server.py for video processing).
mkdir -p "$PROJECT_ROOT/backend_temp"
success "Backend temp directory ready."

echo ""

# ── Step B — Start FastAPI backend server ────────────────────────────────────
info "${BOLD}Step B:${NC} Starting FastAPI backend on http://localhost:8000..."

uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info &
BACKEND_PID=$!
success "FastAPI server running (PID: $BACKEND_PID)"

# Give the server a moment to boot.
sleep 2

# Quick health-check.
if curl -sf http://localhost:8000/ > /dev/null 2>&1; then
    success "Backend health-check passed."
else
    warn "Backend health-check failed — it may still be starting.  Check logs above."
fi

echo ""

# ── Step C — Frontend dev server ─────────────────────────────────────────────
info "${BOLD}Step C:${NC} Installing frontend dependencies and starting Vite dev server..."

if [ ! -d "$PROJECT_ROOT/node_modules" ]; then
    info "Installing npm packages..."
    npm install
    success "npm packages installed."
else
    success "node_modules already exists (run 'npm install' manually if needed)."
fi

# Start Vite in the background.
npm run dev &
FRONTEND_PID=$!
success "Vite dev server starting (PID: $FRONTEND_PID)"

# Give Vite a moment to print its URL.
sleep 3

echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔═══════════════════════════════════════════════════╗"
echo "  ║   🚀 FootballIQ Fully Online                      ║"
echo "  ║                                                   ║"
echo "  ║   Frontend   →  http://localhost:5173              ║"
echo "  ║   Backend    →  http://localhost:8000              ║"
echo "  ║   API Docs   →  http://localhost:8000/docs         ║"
echo "  ║                                                   ║"
echo "  ║   Press Ctrl+C to stop all services.               ║"
echo "  ╚═══════════════════════════════════════════════════╝"
echo -e "${NC}"

# Wait for background processes; Ctrl+C triggers the cleanup trap above.
wait