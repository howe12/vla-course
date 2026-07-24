# 第 4 章：OpenVLA 实战

> 7B 开源 VLA 模型的推理、适配与理解。

---

> **🔄 从 §3 的 L40 云端环境出发** GPU 已就绪、tmux 已学会。现在进入 VLA 推理实战------加载 OpenVLA 7B，让大模型看懂场景、理解指令、输出可以完成任务的机械臂动作。这也是你第一次看到 VLA 模型真正控制机器人。

OpenVLA 实战
------------

7B 开源 VLA 模型的推理、适配与理解。

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
