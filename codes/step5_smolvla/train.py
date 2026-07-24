#!/usr/bin/env python3
"""Step 5 实验：SmolVLA 450M 完整训练脚本

在 L40 上训练 SmolVLA — 从数据加载到模型保存的完整流程。
基于 HuggingFace LeRobot / SmolVLA 架构。

用法（L40 上）：
    # 先准备数据（下载 BridgeData V2 或用自己的数据）
    # 然后：
    uv run python codes/step5_smolvla/train.py

训练时间：~6 小时 (L40, batch=64, 100K steps)
训练成本：~$5
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler  # noqa: F401 (kept for compat)
except ImportError:
    print("❌ PyTorch 未安装")
    sys.exit(1)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 训练
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_steps": 100_000,
    "gradient_accumulation": 1,
    # 模型
    "image_size": 224,
    "num_bins": 256,
    "action_dim": 7,
    "vision_dim": 512,
    "llm_dim": 1024,
    # 日志
    "log_interval": 100,
    "save_interval": 5000,
    "eval_interval": 5000,
    # 数据
    "data_dir": "/root/gpufree-data/bridge_data_v2",
    "output_dir": "outputs/smolvla",
    "num_workers": 8,
}


# ============================================================
# 数据集
# ============================================================
class VLADataset(Dataset):
    """VLA 训练数据集 — 图像 + 指令 + 动作三元组"""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        num_bins: int = 256,
        action_dim: int = 7,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.samples = self._load_index()

    def _load_index(self):
        """加载 JSON 索引文件"""
        index_path = self.data_dir / f"{self.split}_index.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)

        # 如果没有索引文件，扫描目录构建
        print(f"  ⚠️  未找到 {index_path}，尝试扫描目录...")
        split_dir = self.data_dir / self.split
        samples = []
        if split_dir.exists():
            for ep_dir in sorted(split_dir.iterdir()):
                if ep_dir.is_dir():
                    for sample_file in sorted(ep_dir.glob("*.json")):
                        samples.append(str(sample_file))
        print(f"  找到 {len(samples)} 个样本")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]

        # 如果是文件路径字符串，直接加载
        if isinstance(sample_path, str):
            sample_path = Path(sample_path)

        with open(sample_path) as f:
            sample = json.load(f)

        # 加载图像
        if isinstance(sample, dict):
            image_path = sample.get("image_path", "")
            instruction = sample.get("instruction", "")
            action = np.array(sample.get("action", [0] * self.action_dim), dtype=np.float32)
        else:
            # 兼容列表格式
            image_path = str(sample[0]) if len(sample) > 0 else ""
            instruction = str(sample[1]) if len(sample) > 1 else ""
            action = np.array(sample[2] if len(sample) > 2 else [0] * self.action_dim, dtype=np.float32)

        # 加载并预处理图像
        try:
            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")
            else:
                # 占位图（数据缺失时，实际训练应替换）
                image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0

        # 离散化动作: [-1, 1] → [0, 255]
        action = np.clip(action, -1.0, 1.0)
        action_tokens = ((action + 1.0) / 2.0 * (self.num_bins - 1)).astype(np.int64)
        action_tokens = np.clip(action_tokens, 0, self.num_bins - 1)

        return {
            "image": torch.from_numpy(image),
            "action_tokens": torch.from_numpy(action_tokens),
            "instruction": instruction,
        }


# ============================================================
# 模型
# ============================================================
class SmolVLA(nn.Module):
    """SmolVLA: EfficientViT + SmolLM + 7x256 Action Head

    简化版实现用于教学。实际训练建议使用 HuggingFace LeRobot 的
    预训练权重做初始化。
    """

    def __init__(
        self,
        vision_dim: int = 512,
        llm_dim: int = 1024,
        num_bins: int = 256,
        action_dim: int = 7,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.action_dim = action_dim

        # 视觉编码器（教学用简化版，实际用 timm.create_model("efficientvit_b1")）
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, vision_dim, 3, 2, 1),
            nn.AdaptiveAvgPool2d((14, 14)),
        )  # output: (B, vision_dim, 14, 14)

        # 投影层: vision_dim → llm_dim
        self.projector = nn.Linear(vision_dim, llm_dim)

        # 语言基座（简化版 SmolLM，12 层 Transformer Decoder）
        self.llm = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=llm_dim,
                nhead=16,
                dim_feedforward=4096,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=12,  # 教学用 12 层（完整 SmolLM 是 24 层）
        )

        # 动作预测头: 7 个独立的 256 类分类器
        self.action_heads = nn.ModuleList(
            [nn.Linear(llm_dim, num_bins) for _ in range(action_dim)]
        )

    def forward(self, images, action_tokens=None):
        """
        Args:
            images: (B, 3, H, W) RGB 图像
            action_tokens: (B, action_dim) 离散动作标签（训练时）
        Returns:
            训练时: scalar loss
            推理时: (B, action_dim) 连续动作
        """
        B = images.shape[0]

        # 1. 视觉编码
        vis_feat = self.vision_encoder(images)  # (B, vision_dim, 14, 14)
        vis_feat = vis_feat.flatten(2).transpose(1, 2)  # (B, 196, vision_dim)
        vis_tokens = self.projector(vis_feat)  # (B, 196, llm_dim)

        # 2. LLM 前向
        # 用视觉 token 同时做 src 和 tgt（简化版自注意力）
        llm_out = self.llm(vis_tokens, vis_tokens)  # (B, 196, llm_dim)

        # 全局池化：取所有 token 的均值作为动作预测的上下文
        pooled = llm_out.mean(dim=1)  # (B, llm_dim)

        # 3. 动作预测
        action_logits = [head(pooled) for head in self.action_heads]  # 7 x (B, 256)

        if action_tokens is not None:
            # 训练模式：计算交叉熵损失
            losses = []
            for dim_idx, logits in enumerate(action_logits):
                target = action_tokens[:, dim_idx].long()  # (B,)
                loss = nn.CrossEntropyLoss()(logits, target)
                losses.append(loss)
            return torch.stack(losses).mean()
        else:
            # 推理模式：选概率最高的桶 → 连续值
            actions = []
            for logits in action_logits:
                pred_bin = logits.argmax(dim=-1).float()  # (B,)
                # 桶索引 → [-1, 1]
                action_val = pred_bin / (self.num_bins - 1) * 2 - 1
                actions.append(action_val.unsqueeze(-1))
            return torch.cat(actions, dim=-1)  # (B, 7)


# ============================================================
# 训练循环
# ============================================================
def evaluate(model, loader, device):
    """在验证集上计算 loss"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            action_tokens = batch["action_tokens"].to(device)
            with torch.amp.autocast("cuda"):
                loss = model(images, action_tokens=action_tokens)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("❌ 需要 CUDA GPU 才能训练 SmolVLA")
        print("   请在 L40 云端或配有 NVIDIA GPU 的机器上运行")
        sys.exit(1)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")

    # 自动调整 batch_size（根据可用显存）
    if vram_gb < 10:
        auto_bs = 4
        print(f"  ⚠️  显存较小，自动 batch_size: {CONFIG['batch_size']} → {auto_bs}")
        CONFIG["batch_size"] = auto_bs
    elif vram_gb < 20:
        auto_bs = 16
        print(f"  💡 中等显存，自动 batch_size: {CONFIG['batch_size']} → {auto_bs}")
        CONFIG["batch_size"] = auto_bs

    # ============================================================
    # 数据
    # ============================================================
    print(f"\n加载数据集: {CONFIG['data_dir']}")
    try:
        train_dataset = VLADataset(CONFIG["data_dir"], "train", CONFIG["image_size"])
        val_dataset = VLADataset(CONFIG["data_dir"], "val", CONFIG["image_size"])
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        print(f"\n请确保数据目录存在: {CONFIG['data_dir']}")
        print("数据格式见 Step 5.3 课程内容")
        print("\n快速验证：用随机数据测试训练管线")
        print("  python codes/step5_smolvla/train.py --dry-run")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Steps/epoch: {len(train_loader)}")

    # ============================================================
    # 模型
    # ============================================================
    print("\n初始化 SmolVLA 模型...")
    model = SmolVLA(
        vision_dim=CONFIG["vision_dim"],
        llm_dim=CONFIG["llm_dim"],
        num_bins=CONFIG["num_bins"],
        action_dim=CONFIG["action_dim"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数总量: {n_params / 1e6:.1f}M")
    print(f"  目标: 450M (简化教学版 {n_params / 1e6:.0f}M, 完整版用 LeRobot 预训练权重)")

    # ============================================================
    # 优化器
    # ============================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # 学习率调度: 线性 warmup + 余弦衰减
    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            return step / max(CONFIG["warmup_steps"], 1)
        progress = (step - CONFIG["warmup_steps"]) / max(
            CONFIG["max_steps"] - CONFIG["warmup_steps"], 1
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")  # AMP 混合精度

    # ============================================================
    # 训练
    # ============================================================
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    model.train()

    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"开始训练! max_steps={CONFIG['max_steps']}, batch={CONFIG['batch_size']}")
    print(f"预计时间: ~6h (L40)")
    print(f"{'='*60}\n")

    while global_step < CONFIG["max_steps"]:
        for batch in train_loader:
            images = batch["image"].to(device)
            action_tokens = batch["action_tokens"].to(device)

            # 前向 + 反向
            with torch.amp.autocast("cuda"):
                loss = model(images, action_tokens=action_tokens)
                loss = loss / CONFIG["gradient_accumulation"]

            scaler.scale(loss).backward()

            if (global_step + 1) % CONFIG["gradient_accumulation"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1

            # ---- 日志 ----
            if global_step % CONFIG["log_interval"] == 0:
                elapsed = time.time() - start_time
                sps = global_step / elapsed if elapsed > 0 else 0
                eta_h = (
                    (CONFIG["max_steps"] - global_step) / sps / 3600 if sps > 0 else 0
                )
                print(
                    f"Step {global_step:6d}/{CONFIG['max_steps']} | "
                    f"loss={loss.item() * CONFIG['gradient_accumulation']:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"speed={sps:.1f} step/s | "
                    f"ETA={eta_h:.1f}h"
                )

            # ---- 验证 ----
            if global_step % CONFIG["eval_interval"] == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  >>> Val loss: {val_loss:.4f}", end="")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"{CONFIG['output_dir']}/best.pt")
                    print(f"  ✅ Best!")
                else:
                    print()

            # ---- 保存 checkpoint ----
            if global_step % CONFIG["save_interval"] == 0:
                ckpt_path = f"{CONFIG['output_dir']}/ckpt_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    ckpt_path,
                )
                print(f"  >>> Checkpoint saved: {ckpt_path}")

            if global_step >= CONFIG["max_steps"]:
                break

    # ============================================================
    # 完成
    # ============================================================
    total_time = (time.time() - start_time) / 3600
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"  总耗时:     {total_time:.1f}h")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  模型:       {CONFIG['output_dir']}/best.pt")
    print(f"  ⏭️  下一步:  运行 codes/step5_smolvla/eval_libero.py 评估模型")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmolVLA 训练")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式：只用随机数据验证训练管线是否跑通",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("🔬 干跑模式：用随机数据验证训练管线...\n")
        # 创建临时随机数据
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建假的索引文件
            fake_samples = []
            for i in range(100):
                sample = {
                    "image_path": "",
                    "instruction": f"task {i}",
                    "action": list(np.random.uniform(-1, 1, 7)),
                }
                fake_samples.append(sample)

            train_idx = tmpdir + "/train"
            val_idx = tmpdir + "/val"
            os.makedirs(train_idx, exist_ok=True)
            os.makedirs(val_idx, exist_ok=True)

            # 保存索引
            with open(f"{train_idx}/_index.json", "w") as f:
                json.dump(
                    [{"image_path": "", "instruction": f"t{i}", "action": list(np.random.uniform(-1, 1, 7))} for i in range(80)],
                    f,
                )
            with open(f"{val_idx}/_index.json", "w") as f:
                json.dump(
                    [{"image_path": "", "instruction": f"v{i}", "action": list(np.random.uniform(-1, 1, 7))} for i in range(20)],
                    f,
                )

            # 创建 index 文件
            import shutil
            for split in ["train", "val"]:
                split_dir = Path(tmpdir) / split
                split_dir.mkdir(exist_ok=True)
                index_data = []
                for i in range(80 if split == "train" else 20):
                    sample_file = split_dir / f"sample_{i:04d}.json"
                    sample_data = {
                        "image_path": "",
                        "instruction": f"{split} task {i}",
                        "action": list(np.random.uniform(-1, 1, 7)),
                    }
                    with open(sample_file, "w") as f:
                        json.dump(sample_data, f)
                    index_data.append(str(sample_file))
                with open(Path(tmpdir) / f"{split}_index.json", "w") as f:
                    json.dump(index_data, f)

            CONFIG["data_dir"] = tmpdir
            CONFIG["max_steps"] = 200
            CONFIG["log_interval"] = 20
            CONFIG["eval_interval"] = 100
            CONFIG["save_interval"] = 100
            CONFIG["output_dir"] = "/tmp/smolvla_dry_run"
            train()
    else:
        train()
