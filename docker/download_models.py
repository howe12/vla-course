#!/usr/bin/env python3
"""Build-time model downloader — 拉取 HuggingFace 预训练模型到 Docker 镜像中。

运行时机: docker build (作为 Dockerfile 的 RUN 步骤)
目标:   /workspace/models/

OPENVLA 7B ~28GB 不在此下载，改为运行时按需拉取。
"""

import os
from huggingface_hub import snapshot_download, hf_hub_download

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
MODELS_DIR = os.path.join(WORKSPACE, "models")

def download(name: str, repo_id: str, **kwargs):
    dest = os.path.join(MODELS_DIR, name)
    print(f"\n📥 下载 {name}...")
    try:
        snapshot_download(repo_id, local_dir=dest, **kwargs)
        print(f"✅ {name} → {dest}")
    except Exception as e:
        print(f"⚠️ {name} 失败: {e}")

# ─── ACT PushT (小模型 ~300MB, 本地推理需要) ──────────
download("act_default", "lerobot/act_default")
download("act_pusht", "lerobot/act_pusht")

# ─── SmolVLA 基座 (~2GB) ──────────────────────────────
download("smolvla_base", "lerobot/smolvla", pattern="*.json")  # config only
try:
    ckpt = hf_hub_download(
        "lerobot/smolvla", "pretrained_model/smolvla_base.pt",
        local_dir=os.path.join(MODELS_DIR, "smolvla_base"),
    )
    print(f"✅ smolvla_base ckpt → {ckpt}")
except Exception as e:
    print(f"⚠️ smolvla_base ckpt: {e}")

# ─── OpenVLA 7B (~28GB, 标记为运行时下载) ─────────────
print("\n⚠️ OpenVLA 7B ~28GB — 不在此下载")
print("   容器启动后运行: cd /workspace/codes/step4_openvla && python3 run_openvla_libero.py")

print(f"\n✅ 完成 — 模型目录: {MODELS_DIR}")
for d in sorted(os.listdir(MODELS_DIR)):
    path = os.path.join(MODELS_DIR, d)
    if os.path.isdir(path):
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, files in os.walk(path)
                   for f in files)
        print(f"   {d}/  ({size // 1024 // 1024} MB)")
