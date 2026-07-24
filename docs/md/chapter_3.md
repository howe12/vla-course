# 第 3 章：L40 云端训练环境

> SSH 连接、CUDA 配置、依赖管理——为后续训练备好算力。

---

> **🔄 从 §2 的仿真环境出发** MuJoCo + LIBERO 已就绪。现在把**云端算力**搭好------L40 GPU 是后续所有训练的硬件基础。本章配置 SSH、CUDA、tmux，让 SmolVLA（§5）和 Pi0（§6）的训练"开箱即用"。

L40 云端训练环境
----------------

SSH 连接、CUDA 配置、依赖管理------为后续训练备好算力。

> **🎯 学习目标**
>
> -   SSH 免密登录 L40 云端 GPU 服务器
> -   理解 L40 的硬件配置和适用场景（44GB 显存 / 海外机房）
> -   安装 CUDA Toolkit + PyTorch + 核心依赖
> -   掌握 scp/rsync 文件传输和 tmux 持久会话
> -   跑通 GPU 验证脚本，确认环境可用于后续训练

> **📍 本章定位**
>
> 本章是**算力基础设施**。§5 的 SmolVLA 训练（450M, \~6h）和 §6 的 Pi0 训练（3B, \~20h）都需要 GPU。L40 的 44GB 显存可以同时覆盖这两个模型------装好一次环境，后面两章都不用再碰配置。

<div>

### 3.1 为什么选 L40？

训练 VLA 模型对 GPU 有硬性要求：OpenVLA 7B 的 LoRA 微调需要 20-30GB，SmolVLA 全量训练需要 16-24GB，Pi0 需要 30-40GB。消费级显卡（RTX 3090 24GB / 4060 8GB）只能做推理，训练捉襟见肘。

  GPU        显存    SmolVLA 训练       Pi0 训练            价格
  ---------- ------- ------------------ ------------------- ------------
  **L40**    44 GB   ✅ batch=64, \~6h   ✅ batch=16, \~20h   \~\$0.80/h
  A100       80 GB   ✅ batch=128        ✅ batch=64          \~\$2.50/h
  RTX 3090   24 GB   ⚠️ batch=8         ❌ OOM               自有硬件
  RTX 4060   8 GB    ❌ OOM              ❌ OOM               自有硬件

> **🔑 L40 的三个关键优势**
>
> 1.  **44GB 显存**：正好卡在"够用"的甜点------SmolVLA 可以开大 batch，Pi0 刚好能跑
> 2.  **海外机房**：直连 HuggingFace / GitHub，不用配镜像、不用翻墙，模型下载满速
> 3.  **性价比**：\$0.80/h 是 A100 的三分之一，训一晚上不到 \$10

</div>

<div>

### 3.2 SSH 连接与免密登录

L40 的 SSH 连接信息（已记录在你的记忆中）：

    # SSH 连接
    ssh -p 30334 root@120.209.70.195

    # 设置免密登录（本机执行一次）
    ssh-copy-id -p 30334 root@120.209.70.195

    # 之后直接连，无需密码
    ssh -p 30334 root@120.209.70.195

#### 推荐：配置 SSH config 别名

    # 在本机 ~/.ssh/config 中添加
    Host l40
        HostName 120.209.70.195
        Port 30334
        User root
        IdentityFile ~/.ssh/id_rsa

    # 之后只需
    ssh l40

> **⚠️ ⚠️ L40 是手动开关机**
>
> L40 通过云控制台手动开关机，不使用时记得关机（按小时计费）。开机后需要等 1-2 分钟系统启动。如果 SSH 连不上，先去云控制台确认机器状态。

</div>

<div>

### 3.3 硬件与环境概览

登录 L40 后，先确认硬件配置：

    # 查看 GPU
    nvidia-smi
    # 应输出: NVIDIA L40, 46068 MiB

    # 查看 CUDA 版本
    nvcc --version
    # 应输出: CUDA 12.x

    # 查看磁盘
    # 工程路径: /root/gpufree-data/habitat-lab-edu/（PVC 持久化）
    df -h /root/gpufree-data

L40 的核心硬件配置：

  硬件       规格
  ---------- ------------------------------------------
  **GPU**    NVIDIA L40, 48GB 显存, Ada Lovelace 架构
  **CUDA**   12.x
  **CPU**    Intel Xeon, \~16 vCPU
  **RAM**    60GB+
  **磁盘**   PVC 持久化存储, /root/gpufree-data/

> **📍 PVC 持久化存储**
>
> L40 的 `/root/gpufree-data/` 是 PVC 持久化卷------**关机不会丢数据**。训练 checkpoints、下载的模型权重、实验代码都应该放在这个目录下。但注意：**Git 提交还是要 push**------PVC 是本机的持久化，不是远程备份。

</div>

<div>

### 3.4 Python 环境配置

L40 上已有 Python 3.9 环境和 CUDA。本章用 **uv** 管理项目依赖，与本机 §2 的配置保持一致：

    # L40 上执行（首次配置）
    # 1. 确认 Python
    python3 --version  # Python 3.9.x

    # 2. 安装 uv（如果没有）
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env

    # 3. 创建 vla-course 工程目录
    mkdir -p /root/gpufree-data/vla-course
    cd /root/gpufree-data/vla-course

    # 4. 初始化 uv 项目
    uv init

    # 5. 安装 PyTorch（CUDA 12.x）
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # 6. 安装 VLA 训练核心依赖
    uv pip install transformers accelerate datasets
    uv pip install peft bitsandbytes  # LoRA + 量化

#### 验证 GPU 可用

    uv run python -c "
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    "
    # 预期输出:
    # PyTorch: 2.x.x
    # CUDA available: True
    # GPU: NVIDIA L40
    # VRAM: 44.3 GB

</div>

<div>

### 3.5 文件传输：本机 ↔ L40

课程代码在本机开发，训练在 L40 上跑。需要可靠的文件同步方案：

    # 方式 1：scp（单文件/小目录）
    # 本机 → L40
    scp -P 30334 /develop/vla-course/codes/step5_smolvla/train.py root@120.209.70.195:/root/gpufree-data/vla-course/

    # L40 → 本机（下载 checkpoint）
    scp -P 30334 root@120.209.70.195:/root/gpufree-data/vla-course/ckpt/best.pt /develop/vla-course/ckpt/

    # 方式 2：rsync（大目录、增量同步、推荐）
    # 本机 → L40（只传变化的文件）
    rsync -avz -e "ssh -p 30334" \
      /develop/vla-course/codes/ \
      root@120.209.70.195:/root/gpufree-data/vla-course/codes/

    # L40 → 本机（下载训练产物）
    rsync -avz -e "ssh -p 30334" \
      root@120.209.70.195:/root/gpufree-data/vla-course/outputs/ \
      /develop/vla-course/outputs/

> **🔑 scp vs rsync**
>
> **scp** 适合传单个文件（模型权重、checkpoint）。**rsync** 适合同步整个目录（代码、数据集），只传变化的文件，中断后可续传。课程后续的"本机编辑 → rsync 推送 → L40 训练 → rsync 拉回"工作流基于 rsync。

</div>

<div>

### 3.6 tmux：让训练在后台跑

训练 SmolVLA 需要 \~6 小时，Pi0 需要 \~20 小时。SSH 断开会导致前台进程被杀。**tmux** 让训练会话在后台持久运行：

    # 创建新会话
    ssh l40
    tmux new -s train

    # 在 tmux 里启动训练
    cd /root/gpufree-data/vla-course
    uv run python codes/step5_smolvla/train.py

    # 断开 tmux（训练继续在后台跑）
    Ctrl+B, D

    # 重新连接
    tmux attach -t train

    # 查看所有会话
    tmux ls

#### tmux 常用快捷键

  操作       快捷键
  ---------- ------------------------
  断开会话   `Ctrl+B, D`
  新建窗口   `Ctrl+B, C`
  切换窗口   `Ctrl+B, 0-9`
  上下滚动   `Ctrl+B, [` 然后方向键
  水平分屏   `Ctrl+B, "`
  垂直分屏   `Ctrl+B, %`

</div>

<div>

### 3.7 GPU 监控：实时看显存和利用率

训练中需要监控 GPU 状态------显存是否 OOM、利用率是否打满：

    # 基础监控（实时刷新）
    watch -n 1 nvidia-smi

    # 更详细的监控（推荐安装 gpustat）
    uv pip install gpustat
    gpustat -i 1  # 每秒刷新

    # 输出示例:
    # [0] NVIDIA L40 | 48°C,  85% utilization,  32768 / 46068 MB
    #     pid 12345: python train.py  (32600 MB)


⚠️ 训练中关注

-   **显存 (Memory)**：接近 44GB 红线时减小 batch\_size
-   **利用率 (Utilization)**：应该 \> 80%，如果 \< 50% 可能是数据加载瓶颈
-   **温度 (Temp)**：L40 正常在 50-70°C，超过 80°C 检查散热


<div>

### 3.8 动手实验：初始化 L40 训练环境

> **🧪 🧪 把 L40 环境从头配好**
>
> 1.  **SSH 登录**：`ssh -p 30334 root@120.209.70.195`
> 2.  **创建工程目录**：`mkdir -p /root/gpufree-data/vla-course && cd /root/gpufree-data/vla-course`
> 3.  **初始化 uv**：`uv init && uv pip install torch --index-url https://download.pytorch.org/whl/cu121`
> 4.  **验证 GPU**：`uv run python -c "import torch; print(torch.cuda.get_device_name(0))"`
> 5.  **启动 tmux**：`tmux new -s test`
> 6.  **运行 GPU 压测**：`uv run python codes/step4_l40/gpu_test.py`

展开查看 gpu\_test.py（GPU 基准测试脚本）


    #!/usr/bin/env python3
    """L40 GPU 基准测试 - 验证环境可用、测试算力上限"""
    import torch
    import time

    print("=" * 50)
    print("L40 GPU 基准测试")
    print("=" * 50)

    # 1. 基本信息
    print(f"\
    PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_mem / 1024**3:.1f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"CUDA Cores: {props.multi_processor_count} SM")

    # 2. 矩阵乘法算力测试
    print("\
    --- 矩阵乘法算力测试 ---")
    for size in [1024, 2048, 4096, 8192]:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        
        # 预热
        for _ in range(5):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        for _ in range(20):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tflops = 2 * size**3 * 20 / elapsed / 1e12
        print(f"  {size}x{size} matmul: {elapsed/20*1000:.1f}ms ({tflops:.1f} TFLOPS)")

    # 3. 可用显存测试
    print("\
    --- 显存测试 ---")
    for frac in [0.25, 0.5, 0.75, 0.9]:
        try:
            total = torch.cuda.get_device_properties(0).total_mem
            size = int(total * frac) // 4  # float32
            x = torch.zeros(size, device="cuda")
            used = torch.cuda.memory_allocated() / 1024**3
            print(f"  分配 {frac*100:.0f}% 显存: {used:.1f} GB OK")
            del x
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  分配 {frac*100:.0f}% 显存: OOM")

    print("\
    GPU 基准测试完成！")
    print("L40 环境已就绪，可以开始下一步训练。")

跑通这个实验后，L40 环境就完整配置好了。§5 的 SmolVLA 训练脚本直接 scp 上去就能跑。


<div>

### 📝 本章小结

> **带走这几条**
>
> 1.  **L40 的 44GB 显存**恰好覆盖 SmolVLA (16-24GB) 和 Pi0 (30-40GB) 的训练需求
> 2.  **SSH 免密 + config 别名**：一行 `ssh l40` 连接，不输密码
> 3.  **PVC 持久化**：`/root/gpufree-data/` 关机不丢数据，checkpoint 放这里
> 4.  **uv 统一管理**：本机和 L40 用同一套依赖方案，避免版本差异
> 5.  **rsync + tmux**：代码推送 → tmux 后台训练 → 断开 SSH → 过几小时回来收结果

#### 🤔 思考题

1.  如果把 gpu\_test.py 在 RTX 4060 (8GB) 上跑，哪个显存分配比例会 OOM？
2.  rsync 和 scp 都可以传文件------如果一个 10GB 的模型权重已传到 L40 上，你又微调了一下训练脚本（5KB），用什么命令只更新脚本？
3.  tmux 断开后训练继续跑------但如果你重启了 L40（关机再开机），tmux 会话还在吗？怎么办？

> **🔄 从 §3 的 L40 云端环境出发** GPU 已就绪、tmux 已学会。现在进入 VLA 推理实战------加载 OpenVLA 7B，让大模型看懂场景、理解指令、输出可以完成任务的机械臂动作。这也是你第一次看到 VLA 模型真正控制机器人。

L40 云端训练环境
----------------

SSH 连接、CUDA 配置、依赖管理------为后续训练备好算力。

> **🎯 学习目标**
>
> -   理解 OpenVLA 的架构：SigLIP ViT + LLaMA 7B + Action Head 的分工
> -   从 HuggingFace 下载并加载 OpenVLA 7B 模型（FP16 / INT4）
> -   掌握推理管线：图像预处理 → 多模态前向 → 离散 Token → 连续动作
> -   掌握推理管线：图像预处理 模型前向 动作解码 环境闭环
> -   在 LIBERO-Spatial 上运行完整的 VLA 微调模型推理闭环

> **📍 本章定位**
>
> 本章是**VLA 推理的核心章**。你先学会"用"模型------加载现成的 OpenVLA 权重，理解它如何从像素走到关节角。第 5 章（SmolVLA）和第 6 章（Pi0）会教你"训"模型。

<div>

### 4.1 OpenVLA 是什么？

**OpenVLA**（2024.3）是 Stanford HAI 和 UIUC 联合发布的**首个开源 7B VLA 模型**。它的发布标志着 VLA 从"闭源封闭"走向"开源开放"------权重、代码、推理框架全部公开，Apache 2.0 协议可商用。

  特性             OpenVLA                        RT-2（对比）
  ---------------- ------------------------------ ---------------------------------
  **参数量**       7B                             55B
  **视觉编码器**   SigLIP ViT-SO (384M)           ViT-22B
  **语言基座**     LLaMA 2 7B                     T5-XL + PaLI-X
  **训练数据**     Open X-Embodiment (2M+ 轨迹)   内部数据 (13 台机器人, 17 个月)
  **推理速度**     \~40ms (INT4, 25Hz)            \~530ms (FP16, \~2Hz)
  **开源**         ✅ Apache 2.0                   ❌ 闭源

> **🔑 核心创新**
>
> OpenVLA 证明了**开源 LLM（LLaMA 2）可以直接做 VLA 的基座**，不需要 Google 的闭源模型。7B 参数经过 INT4 量化后仅需 \~5GB 显存，可以在消费级 GPU（RTX 3090/4060）上推理。这也是本课程选它的原因------**你能在自己的硬件上跑起来**。

</div>

<div>

### 4.2 OpenVLA 架构详解

OpenVLA 遵循"视觉编码器 → 语言基座 → 动作头"三阶段范式，与 §1 讲的理论架构完全对应：

    ┌──────────────────────────────────────────────────────┐
    │  📷 RGB 图像 (224×224×3)                              │
    │        ↓                                              │
    │  SigLIP ViT-SO (384M)                                │
    │  将图像切为 14×14 = 196 个 patch，输出 196×1024 特征   │
    │        ↓                                              │
    │  ┌─────────────────────────────────────────┐         │
    │  │  Projector: 1024 → 4096 (LLaMA 维度)    │         │
    │  └─────────────────────────────────────────┘         │
    │        ↓                                              │
    │  ┌─────────────────────────────────────────┐         │
    │  │  LLaMA 2 7B (32 层, 4096 hidden)        │         │
    │  │  · 196 个视觉 token + N 个语言 token    │         │
    │  │  · 自回归生成 8 个动作 token             │         │
    │  └─────────────────────────────────────────┘         │
    │        ↓                                              │
    │  ┌─────────────────────────────────────────┐         │
    │  │  Action Head: 每个 token → 256 类分类   │         │
    │  │  7 个动作维 × 256 桶 → 7 个离散索引     │         │
    │  └─────────────────────────────────────────┘         │
    │        ↓                                              │
    │  🦾 连续动作: (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, g)  │
    └──────────────────────────────────────────────────────┘

#### 三个组件的分工

  组件             模型                        参数量   作用
  ---------------- --------------------------- -------- --------------------------------------
  **视觉编码器**   SigLIP ViT-SO               \~384M   RGB → 196 个 1024 维视觉 token
  **投影层**       Linear(1024→4096)           \~4M     视觉维度 → LLaMA 维度对齐
  **语言基座**     LLaMA 2 7B                  \~6.7B   融合视觉+语言 → 自回归生成动作 token
  **动作头**       7 个独立 Linear(4096→256)   \~7M     每个 token → 256 桶概率 → 连续动作值

</div>

<div>

### 4.3 模型下载与加载

OpenVLA 模型托管在 HuggingFace，约 14GB（FP16）。国内下载可能需要镜像：

    # 方案 A：直接下载（推荐，国外网络或代理）
    pip install transformers>=4.40 torch>=2.0 accelerate

    # 方案 B：HF 镜像（国内网络）
    export HF_ENDPOINT=https://hf-mirror.com
    pip install transformers torch accelerate

    # 方案 C：L40 云端（第 3 章）
    # L40 位于海外机房，直连 HuggingFace，无需镜像

#### 加载模型（FP16）

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_id = "openvla/openvla-7b"

    # FP16 加载（需 ~14GB 显存）
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",          # 自动分配 GPU/CPU
        trust_remote_code=True,     # OpenVLA 有自定义代码
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print("模型加载完成！")

#### INT4 量化加载（消费级 GPU）

    # INT4 量化（需 ~8GB 显存，RTX 4060 / 3090 可用）
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )


⚠️ 硬件要求

-   **FP16**：\~14GB 显存 --- RTX 3090/4090, A100, L40
-   **INT8**：\~8GB 显存 --- RTX 3070/4060, V100
-   **INT4**：\~5GB 显存 --- RTX 2060/3060, T4
-   **CPU**：理论上可行（\~64GB RAM），推理速度 \~2-5 秒/步，不推荐


<div>

### 4.4 推理管线：从像素到关节角

一次完整的 OpenVLA 推理包含四个步骤。理解每一步，你就能在任何机器人平台上适配它：

#### 步骤 1：图像预处理

    from PIL import Image
    import numpy as np

    # LIBERO 返回的 RGB 数组 (224, 224, 3) uint8
    obs = env.reset()  # numpy uint8
    image = Image.fromarray(obs).convert("RGB")

    # 可选：反归一化到 [0,1]（OpenVLA processor 会自动处理）

#### 步骤 2：构建 Prompt

    # OpenVLA 使用特殊格式编码任务指令
    # 格式: "Task: {instruction}. Bot: In: Im"
    instruction = task.language_instruction
    prompt = f"Task: {instruction}. Bot: In: Im"

    # processor 自动将 "Im" 替换为图像 token
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(model.device)

#### 步骤 3：模型前向推理

    with torch.no_grad():
        # 自回归生成 8 个 token（7 动作 + 1 结束符）
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,          # 贪心解码 = 确定性输出
            num_beams=1,
        )

    # 提取新生成的 token
    action_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    # shape: (1, 8)

#### 步骤 4：Token → 连续动作

    # OpenVLA 的 action head 是 7 个 Linear(4096, 256) 分类头
    # generate 的输出 token ID 就是每个维度的桶索引 (0~255)

    def decode_action(action_tokens, num_bins=256):
        """
        将离散 token 恢复为连续动作值
        
        action_tokens: (1, 8) — 最后一个是结束符，忽略
        返回: (7,) numpy array, 范围 [-1, 1]
        """
        token_ids = action_tokens[0, :7].cpu().numpy()  # 取前 7 个
        
        actions = []
        for tid in token_ids:
            # 256 桶均匀分布在 [-1, 1]
            bin_centers = np.linspace(-1.0, 1.0, num_bins)
            action_value = bin_centers[tid % num_bins]
            actions.append(action_value)
        
        return np.array(actions)

> **🔑 理解 token → 动作的映射**
>
> 假设第 0 维（Δx）的 token ID = 150，256 桶均匀分布在 \[-1, 1\]：桶 150 的中心 = -1.0 + (150/255) × 2.0 = **+0.176**。这个值就是机械臂末端在 x 方向的增量。ID=0 → -1.0（向左最快），ID=255 → +1.0（向右最快），ID=128 → \~0.0（不动）。

</div>

<div>

### 4.5 实操：完整部署与调试过程

理论讲完了，该动手了。本节记录了一次**真实的 VLA 部署过程**------从选模型到跑通 10 个任务，中间经历了 **8 轮调试、5 个关键修复**。这些坑你大概率也会遇到。

#### 4.5.1 模型选择：基础 vs 微调

这是**整个部署过程中最重要的一步**，也是我们踩的第一个坑。

                      基础模型                 微调模型
  ------------------- ------------------------ ---------------------------------------
  **HuggingFace**     `openvla/openvla-7b`     `openvla-7b-finetuned-libero-spatial`
  **大小**            15 GB                    15.1 GB（完整合并权重）
  **适用机器人**      WidowX (BridgeData V2)   Panda (LIBERO 仿真)
  **LIBERO 成功率**   0%（不认识 Panda）       84.7%（官方）

> **⚠️ 第 1 号坑：模型选错 = 5 轮白跑**
>
> OpenVLA 官方页面明确写着：WidowX & Google 机器人是 zero-shot，**Franka Panda 需要微调**。我们一开始用基础模型跑了 5 轮全部 0%，因为模型根本不认识 Panda 机械臂和 LIBERO 场景。

#### 4.5.2 四个必须对齐的配置

  配置项             错误值            正确值                                  为什么
  ------------------ ----------------- --------------------------------------- ----------------------------
  **模型**           `openvla-7b`      `openvla-7b-finetuned-libero-spatial`   微调版才有 LIBERO 知识
  **unnorm\_key**    `"bridge_orig"`   `"libero_spatial"`                      反归一化统计量匹配训练数据
  **图像旋转**       无                `img[::-1, ::-1]` (180)                 训练数据做了 180 旋转！
  **center\_crop**   `False`           `True`                                  训练时用了随机裁剪增强

完整推理脚本：[GitHub](https://github.com/howe12/vla-course/blob/main/codes/step3_openvla/run_openvla_libero.py)

</div>

<div>

### 4.6 调试复盘：8 轮尝试的教训

这是本章**最有价值的部分**------8 轮失败的完整记录。

  轮次    改动                        结果       根因
  ------- --------------------------- ---------- -----------------------
  **1**   基础模型 + bridge\_orig     0%         模型不认识 Panda
  **2**   换 fractal unnorm           0%         统计量不匹配
  **3**   OSC\_POSE 控制器            0%         控制器对，动作不对
  **4**   OffScreenRenderEnv + 旋转   0%         仍缺微调模型
  **5**   添加 OSC 归一化             0%         误加：动作已 OSC 范围
  **6**   微调模型 + 移除归一化       0%         缺 180 旋转！
  **7**   恢复 OSC 归一化             0%         截断 + 缺旋转
  **8**   全部对齐官方管道            **100%**   微调模型 + 正确配置

#### 三大致命陷阱

> **陷阱 1：图像预处理 = 模型的眼睛**
>
> 官方 `get_libero_image()` 做了 `img[::-1, ::-1]`（180 旋转）。**移除旋转后模型看到颠倒的世界**------碗在左边，它以为在右边。

> **陷阱 2：动作空间误解**
>
> LIBERO 训练数据的动作 **已**在 OSC \[-1,+1\] 范围。`predict_action` 输出可直接用，**不需要**再归一化。

> **陷阱 3：基础设施**
>
> L40 PVC 50GB，两个模型 30GB。必须删旧下新。国内需 HF 镜像。离线部署需修复 `auto_map`。

</div>

<div>

### 4.7 最终成果：10/10 全部通过

所有修复就位后，L40 GPU 上 **257 秒**完成 10 个 LIBERO-Spatial 任务：

    ============================================================
    评估汇总
    ============================================================
      总任务数: 10      成功: 10      失败: 0
      成功率:   100%    总耗时:   257s

#### Task 0 推理过程


![Step 0: 初始场景](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task00_step000.png)

![Step 40: 接近目标](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task00_step040.png)

![Step 79: 任务完成](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task00_step079.png)

#### 更多任务场景


![Task 3: 碗在饼干盒上](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task03_step000.png)

![Task 7: 碗在炉子上](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task07_step000.png)

![Task 9: 碗在木柜上](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task09_step000.png)


<div>

### 4.8 部署检查清单

> **每次部署前逐项确认**
>
> 1.  模型是基础版还是微调版？
> 2.  `unnorm_key` 与 `model.norm_stats` 匹配？
> 3.  图像预处理与训练一致？
> 4.  动作范围：\[-1,+1\] 还是物理值？
> 5.  夹爪约定正确？
> 6.  `trust_remote_code=True`？
> 7.  磁盘空间足够？
> 8.  Python 3.10？

</div>

<div>

### 4.9 理解输出：动作的 token 编码

LLaMA 输出的 token ID 和 256 桶索引的关系：

1.  LLaMA 词表有 32000 个文本 token
2.  OpenVLA 额外加了 **7x256 = 1792 个动作 token**
3.  token ID = 32000 + dim x 256 + bucket
4.  商 = 维度索引，余数 = 桶索引

  Token ID   维度    桶    连续值
  ---------- ------- ----- --------
  32000      0 (x)   0     -1.0
  32128      0 (x)   128   \~0.0
  32255      0 (x)   255   +1.0

</div>

<div>

### 本章小结

> **带走这几条**
>
> 1.  **OpenVLA = SigLIP ViT + LLaMA 2 7B + 7x256 动作头**
> 2.  **基础模型 微调模型**：Panda 机器人必须用 LIBERO 微调版
> 3.  **三对齐**：unnorm\_key 对齐数据、图像预处理对齐训练、动作空间对齐控制器
> 4.  **8 轮调试教会我们**：一次只改一个变量，先对齐官方管道再调参

#### 思考题

1.  为什么基础模型在 LIBERO 上成功率为 0%？如果换一个也用 Panda 臂的数据集微调，能直接迁移吗？
2.  180 图像旋转对模型推理的影响有多大？设计实验验证。
3.  如果要把 OpenVLA 部署到 Gemini 真机上（7），需要改哪些配置？

</div>


</div>


</div>

</div>

</div>


</div>
