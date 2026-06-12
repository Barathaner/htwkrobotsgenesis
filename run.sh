#!/usr/bin/env bash
# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Booster K1 Locomotion (Python 3.12 + Genesis + rsl-rl)          ║
# ╚═══════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"

# ── Colors ────────────────────────────────────────────────────────
BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'

log()  { echo -e "${BLUE}[K1-3.12]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Venv management with Python 3.12 ─────────────────────────────
ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Searching for Python 3.12..."

        if command -v python3.12 &> /dev/null; then
            PYTHON_EXE="python3.12"
        else
            PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
            if [ "$PY_VERSION" = "3.12" ]; then
                PYTHON_EXE="python3"
            else
                err "Python 3.12 is required but not found. Please install it (e.g., 'sudo apt install python3.12 python3.12-venv')."
            fi
        fi

        log "Creating virtual environment using $PYTHON_EXE..."
        $PYTHON_EXE -m venv "$VENV_DIR"
        log "Venv created at $VENV_DIR"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
}

install_deps() {
    ensure_venv
    log "Updating pip..."
    pip install --upgrade pip --quiet

    log "Installing PyTorch (CUDA 12.1)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

    log "Installing Genesis and dependencies..."
    pip install genesis-world mujoco numpy scipy wandb pyyaml --quiet

    log "Installing rsl-rl (PPO)..."
    pip install "rsl-rl-lib>=5.0.0" --quiet

    log "All dependencies ready. PyTorch $(python3 -c 'import torch; print(torch.__version__)') detected."
}

# ── Commands ─────────────────────────────────────────────────────
case "${1:-help}" in
    setup)
        log "Initializing K1 locomotion project..."
        install_deps
        mkdir -p logs models/K1
        log "Setup finished. Use './run.sh shell' or './run.sh train ...'."
        ;;
    train)
        shift
        ensure_venv
        log "Starting K1 PPO training..."
        python3 walk/K1_train.py "$@"
        ;;
    eval)
        shift
        ensure_venv
        python3 walk/K1_eval.py "$@"
        ;;
    test)
        shift
        ensure_venv
        python3 walk/test_k1_env.py "$@"
        ;;
    shell)
        ensure_venv
        log "Environment activated (Python $(python3 --version))."
        exec "$SHELL"
        ;;
    help|--help|-h)
        cat <<EOF
Usage: $0 {setup|train|eval|test|shell}

  setup   Create .venv and install dependencies
  train   Run PPO training  (e.g. $0 train -e k1-walking -B 2048 --max_iterations 500)
  eval    Evaluate a checkpoint
  test    Run K1_env smoke tests  (e.g. $0 test --test all)
  shell   Activate venv and open a subshell
EOF
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: $0 {setup|train|eval|test|shell|help}"
        exit 1
        ;;
esac
