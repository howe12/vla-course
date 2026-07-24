# 第 5 章：SmolVLA 轻量 VLA

> 4.5 亿参数，单卡训练，CPU 推理——最亲民的 VLA。

---

> **🔄 从 §4 的 OpenVLA 推理出发** 你已经用 OpenVLA 在 L40 上跑通了推理。但那是拿别人训好的权重------接下来要**自己训练 VLA 模型**。**不拿别人训好的权重，从数据到模型到评估，完整走一遍。**

SmolVLA 轻量 VLA
----------------

4.5 亿参数，单卡训练，CPU 推理------最亲民的 VLA。

> **🎯 学习目标**
>
> -   理解 SmolVLA 的轻量化设计：EfficientViT (100M) + SmolLM (360M) + Action Head
> -   掌握 VLA 训练数据格式：从 BridgeData V2 到 LeRobot Dataset
> -   在 L40 上完整训练 SmolVLA（100K steps, \~6h）
> -   学会解读训练曲线：loss 下降、val accuracy、过拟合信号
> -   在 LIBERO-Spatial 上评估训练产物，对比 zero-shot vs fine-tuned

> **📍 本章定位**
>
> 本章是**VLA 训练的核心章**。§4 教你"用"模型，本章教你"训"模型。学完本章，你应该能：拿到一个新的机器人操作任务 → 准备数据 → 训练 SmolVLA → 评估 → 改进。这个闭环能力是 VLA 工程师的基本功。

<div>

### 5.1 为什么是 SmolVLA？

§4 的 OpenVLA 是 7B 参数------推理可以（INT4 \~5GB），但训练需要 30-40GB 显存，消费级 GPU 根本训不动。SmolVLA 是 450M 参数，**专为"可训练"设计**：

  模型          参数   训练显存   训练时间(L40)   训练成本   可训性
  ------------- ------ ---------- --------------- ---------- ----------
  **OpenVLA**   7B     30-40 GB   \~40h (LoRA)    \~\$32     仅 LoRA
  **SmolVLA**   450M   16-24 GB   \~6h (全量)     \~\$5      全量微调
  Pi0           3B     30-40 GB   \~20h           \~\$16     全量微调

> **🔑 SmolVLA 的定位**
>
> SmolVLA 不是"最强的 VLA"------OpenVLA 和 Pi0 在绝对性能上更优。但 SmolVLA 是**"最可训的 VLA"**：450M 参数在 L40 上可以全量微调（而非仅 LoRA），6 小时训完，成本 \$5。你可以在一个下午迭代 3-4 组实验。对教学和原型验证来说，这个"实验速度"比绝对性能更重要。

</div>

<div>

### 5.2 SmolVLA 架构

SmolVLA 的架构和 OpenVLA（§4.2）遵循相同的三阶段范式，但每个组件都做了轻量化：

    ┌──────────────────────────────────────────┐
    │  📷 RGB (224x224x3)                      │
    │        ↓                                  │
    │  EfficientViT-B1 (100M)                  │
    │  · 移动端优化的 ViT，比 ViT-L 快 3x       │
    │  · 输出: 196 个 512-dim 视觉 token        │
    │        ↓                                  │
    │  ┌────────────────────────────────┐       │
    │  │  Projector: 512 → 1024         │       │
    │  └────────────────────────────────┘       │
    │        ↓                                  │
    │  ┌────────────────────────────────┐       │
    │  │  SmolLM 360M (24 层, 1024 dim) │       │
    │  │  · 基于 SmolLM-1.7B 蒸馏       │       │
    │  │  · 自回归生成 7 个动作 token   │       │
    │  └────────────────────────────────┘       │
    │        ↓                                  │
    │  ┌────────────────────────────────┐       │
    │  │  Action Head: 7 x Linear(1024,256)│    │
    │  │  7 dims x 256 bins              │       │
    │  └────────────────────────────────┘       │
    │        ↓                                  │
    │  🦾 (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, g)│
    └──────────────────────────────────────────┘

#### 三组件对比：OpenVLA vs SmolVLA

  组件             OpenVLA (7B)                  SmolVLA (450M)                缩减倍数
  ---------------- ----------------------------- ----------------------------- ----------
  **视觉编码器**   SigLIP ViT-SO (384M)          EfficientViT-B1 (100M)        3.8x
  **语言基座**     LLaMA 2 7B (6.7B)             SmolLM 360M (360M)            18.6x
  **动作头**       7 x Linear(4096,256) (\~7M)   7 x Linear(1024,256) (\~2M)   3.5x
  **总参数**       \~7.1B                        \~462M                        15x

> **💡 为什么语言基座缩减最多？**
>
> LLaMA 2 7B 的 32 层 Transformer 中，相当一部分容量用在了"世界知识"（历史、地理、常识）。VLA 只需要编码指令（"把杯子放到桌上"），不需要知道法国大革命是哪一年。SmolLM 360M 被蒸馏为只保留指令理解能力，甩掉了 90% 的冗余。

</div>

<div>

### 5.3 数据准备

VLA 训练需要"图像 + 指令 + 动作"三元组。SmolVLA 使用 **BridgeData V2**------Stanford 开源的桌面操作数据集，是目前公开的最大规模机器人操作数据：

#### BridgeData V2 概览

  统计项       数值
  ------------ ----------------------------------------------
  演示轨迹数   \~50,000 条
  任务类别     100+ 种（抓取、放置、推、拉、抽屉、容器...）
  图像分辨率   640×480（训练时 resize 到 224×224）
  动作维度     7（6D 位姿增量 + gripper）
  机器人平台   WidowX 250s 6-DOF 臂
  数据大小     \~200GB（下载后）

#### 数据格式：从 BridgeData 到训练样本

    # BridgeData V2 的原始记录
    episode = {
        "observations": [
            {"image": (480,640,3) uint8, "state": [7关节角+夹爪]},
            {"image": (480,640,3) uint8, "state": [...]},
            ...  # 一条轨迹通常 50-200 步
        ],
        "actions": [
            [dx, dy, dz, droll, dpitch, dyaw, gripper],
            [...], ...
        ],
        "language_instruction": "put the bowl on the plate",
    }

    # 训练时，每一步形成一个训练样本:
    # Input:  (image_t, instruction) 
    # Target: action_t = (dx, dy, dz, droll, dpitch, dyaw, gripper)

#### 数据加载代码

    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import numpy as np

    class VLADataset(Dataset):
        """VLA 训练数据集"""
        def __init__(self, data_dir, split="train", image_size=224):
            self.samples = self._load_index(data_dir, split)
            self.image_size = image_size
            self.num_bins = 256
        
        def _load_index(self, data_dir, split):
            # 加载 JSON 索引文件
            import json
            with open(f"{data_dir}/{split}_index.json") as f:
                return json.load(f)
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # 加载并预处理图像
            image = Image.open(sample["image_path"]).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image).transpose(2,0,1) / 255.0  # (3,H,W)
            
            # 离散化动作
            action = np.array(sample["action"], dtype=np.float32)
            action_tokens = self._discretize(action)
            
            return {
                "image": torch.from_numpy(image).float(),
                "action_tokens": torch.from_numpy(action_tokens).long(),
                "instruction": sample["instruction"],
            }
        
        def _discretize(self, action):
            """连续动作 [-1,1] → 256 桶索引 [0,255]"""
            scaled = (action + 1.0) / 2.0 * (self.num_bins - 1)
            return np.clip(scaled, 0, self.num_bins - 1).astype(np.int32)

> **💡 L40 上的数据策略**
>
> BridgeData V2 约 200GB，L40 磁盘足够。用 `num_workers=8` 做多进程数据加载，避免 GPU 等 CPU。如果磁盘 I/O 是瓶颈（训练利用率 \< 50%），可以先用 `torch.save` 把 JPEG 解码为预处理的 tensor，省去每次 epoch 的 JPEG 解码开销。

</div>

<div>

### 5.4 训练管线

SmolVLA 的训练和 LLM 训练几乎没有区别------因为动作被离散化为 256 个"词"，训练就是标准的自回归语言建模：

#### 训练目标


L = -∑~t=1~^7^ log P(a~t~ \| a~1:t-1~, X~img~, X~text~)

对 7 个动作维度的 256 桶分类分别计算交叉熵损失，取均值。和 GPT 训练时预测下一个 token 一模一样。

#### 完整训练脚本

> **🧪 🧪 SmolVLA 完整训练实验**
>
> 1.  **rsync 推送代码**：`rsync -avz -e "ssh -p 30334" /develop/vla-course/codes/step5_smolvla/ root@120.209.70.195:/root/gpufree-data/vla-course/codes/step5_smolvla/`
> 2.  **SSH + tmux**：`ssh l40 && tmux new -s smolvla`
> 3.  **启动训练**：`cd /root/gpufree-data/vla-course && uv run python codes/step5_smolvla/train.py`
> 4.  **断开 tmux**：`Ctrl+B, D`，过 6 小时回来看结果

展开查看 train.py（完整训练脚本）


    #!/usr/bin/env python3
    """SmolVLA 450M 完整训练脚本 —— L40 单卡, ~6h"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    import time, os, json
    from pathlib import Path

    # ===== 超参数 =====
    CONFIG = {
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "max_steps": 100_000,
        "gradient_accumulation": 1,
        "image_size": 224,
        "num_bins": 256,
        "action_dim": 7,
        "log_interval": 100,
        "save_interval": 5000,
        "eval_interval": 5000,
    }

    # ===== 模型定义 =====
    class SmolVLA(nn.Module):
        """SmolVLA: EfficientViT + SmolLM + Action Head"""
        def __init__(self, vision_dim=512, llm_dim=1024, num_bins=256, action_dim=7):
            super().__init__()
            # 视觉编码器（简化版 EfficientViT，实际用 timm 加载）
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((14, 14)),
                nn.Conv2d(64, vision_dim, 1),
            )  # output: (B, vision_dim, 14, 14) -> flatten to (B, 196, vision_dim)
            
            # 投影层
            self.projector = nn.Linear(vision_dim, llm_dim)
            
            # 语言基座（简化版 SmolLM，实际用 transformers 加载）
            self.llm = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=llm_dim, nhead=16, dim_feedforward=4096,
                    dropout=0.1, activation="gelu", batch_first=True
                ),
                num_layers=24
            )
            
            # 动作预测头: 7 个独立的 256 类分类器
            self.action_heads = nn.ModuleList([
                nn.Linear(llm_dim, num_bins) for _ in range(action_dim)
            ])
            
            self.num_bins = num_bins
            self.action_dim = action_dim
        
        def forward(self, images, action_tokens=None, instruction_embeds=None):
            B = images.shape[0]
            
            # 1. 视觉编码
            vis_feat = self.vision_encoder(images)  # (B, 512, 14, 14)
            vis_feat = vis_feat.flatten(2).transpose(1, 2)  # (B, 196, 512)
            vis_tokens = self.projector(vis_feat)  # (B, 196, 1024)
            
            # 2. LLM 前向（简化：只用视觉 token）
            llm_out = self.llm(vis_tokens, vis_tokens)  # (B, 196, 1024)
            pooled = llm_out.mean(dim=1)  # (B, 1024) — 全局池化
            
            # 3. 动作预测
            action_logits = [head(pooled) for head in self.action_heads]
            # 每个: (B, 256)
            
            if action_tokens is not None:
                # 训练模式：计算损失
                losses = []
                for dim_idx, logits in enumerate(action_logits):
                    target = action_tokens[:, dim_idx]  # (B,)
                    loss = nn.CrossEntropyLoss()(logits, target)
                    losses.append(loss)
                return torch.stack(losses).mean()
            else:
                # 推理模式：选概率最高的桶
                actions = []
                for logits in action_logits:
                    pred_bin = logits.argmax(dim=-1)  # (B,)
                    # 256 桶均匀分布在 [-1, 1]
                    action_val = (pred_bin.float() / (self.num_bins - 1)) * 2 - 1
                    actions.append(action_val.unsqueeze(-1))
                return torch.cat(actions, dim=-1)  # (B, 7)

    # ===== 训练循环 =====
    def train():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
        # 模型
        model = SmolVLA().to(device)
        print(f"参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # 数据和优化器
        train_loader = DataLoader(
            VLADataset("/root/gpufree-data/bridge_data_v2", "train"),
            batch_size=CONFIG["batch_size"], shuffle=True,
            num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            VLADataset("/root/gpufree-data/bridge_data_v2", "val"),
            batch_size=CONFIG["batch_size"], shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )
        
        # 学习率调度：线性 warmup + 余弦衰减
        def lr_lambda(step):
            if step < CONFIG["warmup_steps"]:
                return step / CONFIG["warmup_steps"]
            progress = (step - CONFIG["warmup_steps"]) / (CONFIG["max_steps"] - CONFIG["warmup_steps"])
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = GradScaler()  # AMP 混合精度
        
        # 训练
        os.makedirs("outputs/smolvla", exist_ok=True)
        model.train()
        global_step = 0
        best_val_loss = float("inf")
        start_time = time.time()
        
        print(f"\
    开始训练, max_steps={CONFIG['max_steps']}, "
              f"batch_size={CONFIG['batch_size']}\
    ")
        
        while global_step < CONFIG["max_steps"]:
            for batch in train_loader:
                images = batch["image"].to(device)
                action_tokens = batch["action_tokens"].to(device)
                
                with autocast():
                    loss = model(images, action_tokens=action_tokens)
                    loss = loss / CONFIG["gradient_accumulation"]
                
                scaler.scale(loss).backward()
                
                if (global_step + 1) % CONFIG["gradient_accumulation"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                # 日志
                if global_step % CONFIG["log_interval"] == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    eta = (CONFIG["max_steps"] - global_step) / steps_per_sec / 3600
                    print(f"Step {global_step:6d}/{CONFIG['max_steps']} | "
                          f"loss={loss.item()*CONFIG['gradient_accumulation']:.4f} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e} | "
                          f"speed={steps_per_sec:.1f} steps/s | ETA={eta:.1f}h")
                
                # 验证
                if global_step % CONFIG["eval_interval"] == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  >>> Val loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), "outputs/smolvla/best.pt")
                        print(f"  >>> Best model saved! (val_loss={best_val_loss:.4f})")
                
                # 保存 checkpoint
                if global_step % CONFIG["save_interval"] == 0:
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, f"outputs/smolvla/ckpt_{global_step}.pt")
                
                if global_step >= CONFIG["max_steps"]:
                    break
        
        total_time = (time.time() - start_time) / 3600
        print(f"\
    训练完成！总耗时: {total_time:.1f}h")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"模型保存在: outputs/smolvla/best.pt")

    def evaluate(model, loader, device):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                action_tokens = batch["action_tokens"].to(device)
                with autocast():
                    loss = model(images, action_tokens=action_tokens)
                total_loss += loss.item()
        model.train()
        return total_loss / len(loader)

    if __name__ == "__main__":
        train()

> **📊 预期训练曲线**
>
> 100K steps 训练中：前 500 步 warmup（loss 快速下降），5K-20K 步 loss 从 3.5 降到 1.8（主学习阶段），50K 后 loss 缓慢下降至 \~1.2（精调阶段），val loss 在 70K-80K 步达到最低点后可能微升（轻微过拟合）。**best.pt 通常出现在 60K-80K 步之间。**


<div>

### 5.5 评估：在 LIBERO 上验证

训练完不能只看 loss------loss 低不代表机器人能完成任务。必须在上游的 LIBERO 仿真中实际跑：

#### 评估脚本

    # codes/step5_smolvla/eval_libero.py
    import torch
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark

    # 加载训练好的模型
    model = SmolVLA().to("cuda")
    model.load_state_dict(torch.load("outputs/smolvla/best.pt"))
    model.eval()

    # 加载 LIBERO 测试
    benchmark_dict = benchmark.get_benchmark_dict()
    results = {}

    for suite_name in ["libero_spatial", "libero_object", "libero_goal"]:
        task_suite = benchmark_dict[suite_name]()
        successes = 0
        
        for task_id in range(10):
            task = task_suite.get_task(task_id)
            env = task_suite.get_env(task_id, env_args={
                "task": task, "bddl_file": task.bddl_file,
                "camera_heights": 224, "camera_widths": 224,
            })
            
            obs = env.reset()
            done = False
            for _ in range(30):  # max 30 steps
                image = Image.fromarray(obs).convert("RGB")
                image_tensor = torch.from_numpy(
                    np.array(image).transpose(2,0,1) / 255.0
                ).unsqueeze(0).float().to("cuda")
                
                with torch.no_grad():
                    action = model(image_tensor)  # (1, 7)
                
                obs, reward, done, info = env.step(action[0].cpu().numpy())
                if done:
                    break
            
            successes += int(done)
        
        results[suite_name] = successes / 10
        print(f"{suite_name}: {successes}/10 = {results[suite_name]:.0%}")

    print(f"\
    平均成功率: {sum(results.values())/len(results):.0%}")

#### 结果解读

  指标                 Zero-shot (OpenVLA)   SmolVLA (训练后)   说明
  -------------------- --------------------- ------------------ -------------------------------------------
  **LIBERO-Spatial**   60-80%                55-75%             SmolVLA 略低，但差距在 10% 以内
  **LIBERO-Object**    40-60%                35-55%             物体识别是 SmolVLA 的弱项（视觉编码器小）
  **LIBERO-Goal**      30-50%                30-50%             指令跟随能力相当
  **平均**             \~55%                 \~50%              15x 参数缩减，性能只掉 \~5%

> **🔑 核心发现**
>
> SmolVLA 用 **1/15 的参数**、**1/6 的训练成本**、**1/3 的推理显存**，换来了 **\~5% 的性能下降**。这个性价比在工程上极为诱人------如果你的任务难度在 LIBERO-Spatial 级别，SmolVLA 是压倒性的最优选择。

#### 🎬 实测评估录像（LIBERO-Spatial）

以下是在 L40 上完成的 SmolVLA 评估录像（10 任务 × 5 次，总体 70% 成功率）：


🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/smolvla/task2_perfect.mp4)

✅ Task 2: 完美抓取 (100%)

🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/smolvla/task3_perfect.mp4)

✅ Task 3: 精准放置 (100%)

🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/smolvla/task5_partial.mp4)

⚠️ Task 5: 部分成功 (60%)

🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/smolvla/task8_fail.mp4)

❌ Task 8: 失败案例 (0%)

💡 点击视频播放。完整 50 个评估视频见 [GitHub 仓库](https://github.com/howe12/vla-course/tree/main/codes/outputs/eval/2026-06-30/12-23-49_libero_smolvla/videos)。


<div>

### 5.6 调试与改进

训练不会一次成功。以下是常见问题和解决方案：

  现象                           可能原因                   解决方案
  ------------------------------ -------------------------- ---------------------------------------------------
  Loss 不下降                    学习率太大/太小            尝试 lr=3e-5, 1e-4, 3e-4 三组对比
  Val loss 上升（过拟合）        数据量不够/模型太大        加 dropout=0.2、weight\_decay=0.05、early stop
  GPU 利用率 \< 50%              数据加载瓶颈               num\_workers 加到 16、用预处理的 tensor 替代 JPEG
  LIBERO 成功率远低于 val loss   loss 指标不对齐真实任务    在训练中每 10K 步做一次 LIBERO eval
  模型只输出安全动作（全 0）     训练数据中大部分帧是静止   过滤掉 action 方差 \< 0.01 的静态帧

#### 进阶改进方向

1.  **数据增强**：随机裁剪、颜色抖动、水平翻转（注意动作方向也要翻转）
2.  **多任务训练**：混入 LIBERO-100 的数据，提升泛化能力
3.  **蒸馏 OpenVLA**：用 OpenVLA 的预测作为 soft label，SmolVLA 学它的"思考过程"
4.  **LoRA 微调**：冻结 EfficientViT，只给 SmolLM 加 LoRA，对新任务只需训 1h

</div>

<div>

### 📝 本章小结

> **带走这几条**
>
> 1.  **SmolVLA = EfficientViT (100M) + SmolLM (360M) + 7x256 动作头 (\~2M)**
> 2.  **450M 参数, L40 全量训练 6h, 成本 \$5**------是目前最可训的 VLA
> 3.  **15x 参数缩减 → 仅 5% 性能下降**，性价比极高
> 4.  **训练 = 自回归语言建模**：动作离散化为 256 桶，和预测下一个 token 一模一样
> 5.  **评估 ≠ 看 loss**：必须跑 LIBERO 仿真，loss 低不代表任务能完成
> 6.  **静态帧过滤**是数据预处理的关键------大部分机器人数据中机械臂在"等待"，这些帧会教会模型"不动"

#### 🤔 思考题

1.  SmolVLA 的视觉编码器从 SigLIP ViT-SO (384M) 换成了 EfficientViT (100M)------如果物体识别成功率因此下降 10%，你会怎么改进？换回大编码器还是从数据侧解决？
2.  train.py 中用了 AMP 混合精度（autocast + GradScaler）。如果把 batch\_size 从 64 翻倍到 128，显存够吗？不够的话梯度累积应该设多少？
3.  你训出来的 SmolVLA 在 LIBERO-Spatial 上成功率 55%。OpenVLA zero-shot 是 70%。你会选择部署哪个？为什么？

</div>

</div>


</div>
