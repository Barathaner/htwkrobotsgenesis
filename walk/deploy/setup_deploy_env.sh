#!/usr/bin/env bash
# Creates a lightweight Python venv for deploying the K1 walking policy.
# Installs: PyTorch (CPU-only) + Booster Robotics SDK Python bindings.
#
# Usage:
#   bash walk/deploy/setup_deploy_env.sh
#
# The venv is created at <repo_root>/deploy_venv.
# Activate it afterwards with:
#   source deploy_venv/bin/activate
#
# Then run the smoke test with:
#   python walk/deploy/smoke_test_k1.py --model logs/foot-rool-penaly/model_1250.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/deploy_venv"

# Path to the booster_robotics_sdk checkout.
# Adjust this if your SDK lives elsewhere.
SDK_DIR="${BOOSTER_SDK_DIR:-$HOME/Dokumente/git/booster_robotics_sdk}"

echo "========================================"
echo "  K1 Deploy venv setup"
echo "  REPO_ROOT : $REPO_ROOT"
echo "  VENV_DIR  : $VENV_DIR"
echo "  SDK_DIR   : $SDK_DIR"
echo "========================================"

# ── 1. Create venv ──────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at $VENV_DIR — skipping creation."
else
    python3 -m venv "$VENV_DIR"
    echo "Venv created."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 2. PyTorch CPU ──────────────────────────────────────────────────────────
# Install CPU-only torch; lighter and sufficient for policy inference.
# The robot's onboard PC typically does not have a CUDA GPU.
echo "Installing PyTorch (CPU)…"
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
echo "  torch $(python -c 'import torch; print(torch.__version__)')"

# ── 3. Booster Robotics SDK Python bindings ─────────────────────────────────
if ! python -c "import booster_robotics_sdk_python" 2>/dev/null; then
    echo "Building Booster Robotics SDK Python bindings from $SDK_DIR …"

    if [ ! -d "$SDK_DIR" ]; then
        echo "ERROR: SDK not found at $SDK_DIR" >&2
        echo "Set BOOSTER_SDK_DIR=<path> or clone it first:" >&2
        echo "  git clone https://github.com/BoosterRobotics/booster_robotics_sdk $SDK_DIR" >&2
        exit 1
    fi

    # Build-time dependencies for scikit-build-core
    pip install scikit-build-core pybind11 --quiet

    # Install SDK build deps (needs apt on Ubuntu; skip if already present)
    if command -v apt-get &>/dev/null; then
        echo "Installing SDK system dependencies (may require sudo)…"
        sudo apt-get install -y --quiet \
            build-essential cmake libssl-dev libasio-dev libtinyxml2-dev
    fi

    pushd "$SDK_DIR" > /dev/null
    pip install . --quiet
    popd > /dev/null

    echo "  booster_robotics_sdk_python installed."
else
    echo "  booster_robotics_sdk_python already installed."
fi

# ── 4. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Setup complete."
echo ""
echo "  Activate with:"
echo "    source deploy_venv/bin/activate"
echo ""
echo "  Smoke test (robot connected via eth0):"
echo "    python walk/deploy/smoke_test_k1.py \\"
echo "        --model logs/foot-rool-penaly/model_1250.pt \\"
echo "        --interface eth0 \\"
echo "        --duration 10"
echo "========================================"
