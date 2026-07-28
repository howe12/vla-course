#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# VLA Course — AMD ROCm 环境一键部署脚本
# 仓库: github.com/howe12/vla-course (amd-rocm-backup)
# 适用: Ubuntu 22.04/24.04 + AMD Radeon GPU (gfx1100+)
# 用法: bash setup.sh [--skip-rocm] [--skip-weights]
# ═══════════════════════════════════════════════════════════════

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[OK]${NC}  $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC} $1"; exit 1; }

SKIP_ROCM=false; SKIP_WEIGHTS=false
for arg in "$@"; do
    case $arg in --skip-rocm) SKIP_ROCM=true ;; --skip-weights) SKIP_WEIGHTS=true ;; esac
done

echo "============================================"
echo " VLA Course — AMD ROCm Environment Setup"
echo "============================================"
echo ""

# ── Step 1: System packages ───────────────────────────────────
log "Step 1/6: Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq python3.12 python3.12-venv python3-pip git wget curl \
    libglfw3 libglfw3-dev libosmesa6 xvfb ffmpeg tini > /dev/null 2>&1
log "System packages done"

# ── Step 2: ROCm (optional) ───────────────────────────────────
if [ "$SKIP_ROCM" = false ]; then
    log "Step 2/6: Installing ROCm..."
    if command -v rocm-smi &>/dev/null; then
        warn "ROCm already installed, skipping"
    else
        wget -q https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install.deb
        sudo apt install -y -qq ./amdgpu-install.deb > /dev/null 2>&1
        sudo amdgpu-install -y --usecase=rocm > /dev/null 2>&1
        rm -f amdgpu-install.deb
        log "ROCm installed"
    fi
    rocm-smi 2>/dev/null || warn "rocm-smi not found — GPU may not be available"
else
    warn "Skipping ROCm install (--skip-rocm)"
fi

# ── Step 3: Python venv ───────────────────────────────────────
log "Step 3/6: Creating Python 3.12 virtual environment..."
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

# Install PyTorch for ROCm
log "Installing PyTorch (ROCm)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0 -q 2>&1 | tail -1

# Install core deps from requirements.txt if exists, else manual
if [ -f requirements.txt ]; then
    log "Installing from requirements.txt..."
    pip install -r requirements.txt -q
else
    log "Installing core packages..."
    pip install -q \
        lerobot==0.6.0 \
        mujoco==3.10.0 \
        transformers>=4.45.0 \
        accelerate \
        datasets \
        opencv-python-headless \
        numpy \
        pyyaml
fi
log "Python environment ready"

# ── Step 4: Verify installations ──────────────────────────────
log "Step 4/6: Verifying installations..."
python -c "import torch; assert torch.cuda.is_available(), 'GPU not available!'; print(f'  PyTorch {torch.__version__} + ROCm OK')"
python -c "import mujoco; print(f'  MuJoCo {mujoco.__version__} OK')"
python -c "import lerobot; print(f'  LeRobot {lerobot.__version__} OK')"
python -c "import cv2; print(f'  OpenCV {cv2.__version__} OK')"

# ── Step 5: Model weights (optional) ──────────────────────────
if [ "$SKIP_WEIGHTS" = false ]; then
    log "Step 5/6: Downloading model weights..."
    export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

    # SmolVLM tokenizer/processor
    mkdir -p weights/smolvlm2
    python -c "
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
p.save_pretrained('weights/smolvlm2')
" 2>&1 | tail -1
    log "SmolVLM processor downloaded"

    # Datawhale SmolVLA weights
    mkdir -p weights/smolvla_datawhale
    huggingface-cli download Datawhale/every-embodied-smolvla-mujoco-pnp \
        --local-dir weights/smolvla_datawhale 2>&1 | tail -3
    log "SmolVLA weights downloaded"
else
    warn "Skipping weight download (--skip-weights)"
fi

# ── Step 6: Environment config ─────────────────────────────────
log "Step 6/6: Configuring environment..."

# Xvfb virtual display
if ! pgrep -x Xvfb > /dev/null; then
    Xvfb :99 -screen 0 1280x960x24 +extension GLX +render &
    echo $! > /tmp/xvfb.pid
    sleep 1
    log "Xvfb started on :99 (PID $(cat /tmp/xvfb.pid))"
else
    warn "Xvfb already running"
fi
export DISPLAY=:99

# Offline mode (prevent accidental downloads)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Add to .bashrc for persistence
grep -q "DISPLAY=:99" ~/.bashrc 2>/dev/null || {
    cat >> ~/.bashrc << 'EOF'
# VLA Course environment
export DISPLAY=:99
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
EOF
    log "Environment variables added to ~/.bashrc"
}

# Verify IK eps
log "Verifying IK eps setting..."
grep "ik_eps" mujoco_env/ik.py | head -1

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "  Source venv:  source venv/bin/activate"
echo "  Run inference: python scripts/inference/infer_ik_fix.py"
echo "  Output dir:   outputs/course_capture/ik_1e-4/"
echo ""
echo "  IK eps:       $(grep ik_eps mujoco_env/ik.py | head -1 | awk '{print $NF}')"
echo "  GPU:          $(rocm-smi 2>/dev/null | head -6 | tail -1 || echo 'check rocm-smi')"
echo ""
