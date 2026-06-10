#!/usr/bin/env bash
# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Elegoo V4 RL - Navigation (Python 3.12 + Genesis)               ║
# ╚═══════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"

# ── Colors ────────────────────────────────────────────────────────
BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'

log()  { echo -e "${BLUE}[ELEGOO-3.12]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Venv management with Python 3.12 ─────────────────────────────
ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Searching for Python 3.12..."
        
        # Check if python3.12 command exists
        if command -v python3.12 &> /dev/null; then
            PYTHON_EXE="python3.12"
        else
            # Fallback check if 'python3' is 3.12
            PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
            if [ "$PY_VERSION" = "3.12" ]; then
                PYTHON_EXE="python3"
            else
                err "Python 3.12 is required but not found. Please install it (e.g., 'sudo apt install python3.12')."
            fi
        fi

        log "Creating virtual environment using $PYTHON_EXE..."
        $PYTHON_EXE -m venv "$VENV_DIR"
        log "Venv created successfully."
    fi
    
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
}

install_deps() {
    ensure_venv
    log "Updating pip..."
    pip install --upgrade pip --quiet

    # 1. ZUERST PyTorch installieren (Wichtig für Python 3.12 & Genesis)
    log "Installing PyTorch (CUDA optimized)..."
    # Hier wird explizit die Version für CUDA 12.1 geladen (Standard für moderne GPUs)
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

    # 2. DANACH Genesis und den Rest
    log "Installing Genesis and dependencies..."
    pip install genesis-world mujoco numpy scipy wandb --quiet

    log "All dependencies ready. PyTorch $(python3 -c 'import torch; print(torch.__version__)') detected."
}

# ── Commands ─────────────────────────────────────────────────────
case "${1:-help}" in
    setup)
        log "Initializing Elegoo Nav Project..."
        install_deps
        mkdir -p models/robot models/flat checkpoints logs
        log "Setup finished. Use './scripts/run.sh shell' to enter the environment."
        ;;
    train)
        shift
        ensure_venv
        log "Starting Python 3.12 Training Loop..."
        python3 -m training.train_nav "$@"
        ;;
    debug)
        shift
        ensure_venv
        python3 -m scripts.debug_scene "$@"
        ;;
    shell)
        ensure_venv
        log "Environment activated (Python $(python3 --version))."
        exec "$SHELL"
        ;;
    *)
        echo "Usage: $0 {setup|train|debug|shell}"
        exit 1
        ;;
esac