FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="NXROBO VLA Course"
LABEL description="VLA Course Docker — LeRobot + ACT/SmolVLA/Pi0 + Gemini"

# ─── System deps ───────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git curl wget ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libopenblas0 \
    ffmpeg libsndfile1 \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# ─── Python setup ──────────────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip setuptools wheel

# ─── PyTorch (CUDA 12.8) ───────────────────────────────
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ─── LeRobot + course deps ─────────────────────────────
RUN pip3 install \
    lerobot>=0.6.1 \
    huggingface_hub \
    datasets \
    accelerate \
    diffusers \
    transformers \
    einops \
    safetensors \
    draccus \
    gymnasium \
    opencv-python-headless \
    av \
    imageio \
    pyyaml \
    tqdm \
    wandb \
    tensorboard

# ─── Workspace ─────────────────────────────────────────
ENV WORKSPACE=/workspace
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
RUN mkdir -p ${WORKSPACE}/models ${WORKSPACE}/outputs ${WORKSPACE}/ckpt ${HF_HOME} ${TORCH_HOME}

# ─── Download HuggingFace models (build-time) ──────────
COPY docker/download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py && rm /tmp/download_models.py

# ─── Course codes ──────────────────────────────────────
COPY codes/ ${WORKSPACE}/codes/
COPY scripts/ ${WORKSPACE}/scripts/
COPY requirements.txt ${WORKSPACE}/requirements.txt 2>/dev/null || true

# ─── Runtime ───────────────────────────────────────────
WORKDIR ${WORKSPACE}
EXPOSE 8888 6006
CMD ["/bin/bash"]
