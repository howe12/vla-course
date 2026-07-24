# VLA 从零到部署 — 具身智能教科书

> **线上阅读:** [howe12.github.io/vla-course](https://howe12.github.io/vla-course)  
> **仓库:** [github.com/howe12/vla-course](https://github.com/howe12/vla-course)  
> **前置课程:** VLN（视觉语言导航）→ 模仿学习 → **VLA（本课程）**

VLA（Vision-Language-Action）是具身智能的核心技术——用一个统一的大模型同时理解语言、看懂图像、输出机器人动作。本课程带你从理论出发，经仿真验证、云端训练，最终部署到真实机器人。

---

## 课程大纲

| 章节 | 名称 | 目的 | 实操内容 | 子节数 | 状态 |
|------|------|------|----------|--------|------|
| **Step 1** | VLA 基础与架构 | 建立 VLA 理论地基——什么是 VLA、和 VLN/VLM/模仿学习的区别 | 无（纯理论） | 11 节 + 小结 + 4 思考题 | ✅ |
| **Step 2** | 仿真环境搭建 | 搭建实验基础设施，学生第一次看到机械臂在仿真中动起来 | `sim_vla_arm.py`（MuJoCo Viewer 实时动画）<br>`run_libero_demo.py`（LIBERO 验证） | 6 节 + 模型文件 + 2 脚本 | ✅ |
| **Step 3** | OpenVLA 实战 | 用现成 7B VLA 模型跑通完整推理管线——从像素到关节角 | `run_openvla_libero.py`（OpenVLA + LIBERO 零样本推理闭环） | 6 节 + 1 脚本 | ✅ |
| **Step 4** | L40 云端训练环境 | 为后续训练备好算力——SSH、CUDA、文件传输、持久会话 | `gpu_test.py`（GPU 基准测试：TFLOPS + 显存分配） | 8 节 + 1 脚本 | ✅ |
| **Step 5** | SmolVLA 轻量 VLA ⭐ | 从数据到模型到评估，完整走一遍 VLA 训练闭环 | `train.py`（完整训练含 --dry-run）<br>`eval_libero.py`（三套件评估） | 6 节 + 2 脚本 | ✅ |
| **Step 6** | Pi0 扩散 VLA ⭐ | 理解 VLA 第二范式——扩散模型为什么比自回归更适合机器人动作 | `sim_pi0_arm.py`（Pi0/Mock 双模式驱动课程机械臂） | 6 节 + 1 脚本 | ✅ |
| **Step 7** | Gemini 真机部署 | 仿真到真机的最后一公里——L40 模型部署到 Gemini 双臂机器人 | `deploy_model.py`（模型转换）<br>`run_gemini.py`（真机推理） | 6 节（计划） | ⏳ |
| **Step 8** | VLA 前沿与竞赛 | 站在前沿看方向——帮学生找到继续深入的路 | `acot_vla_demo.py`（ACoT 推理链可视化） | 4 节（计划） | ⏳ |

---

## 各章详解

### Step 1 — VLA 基础与架构

| 子节 | 内容 |
|------|------|
| 1.1 | VLA 定义 + 数学公式（自回归动作生成） |
| 1.2 | 四范式对比（VLM / VLN / 模仿学习 / VLA） |
| 1.3 | 发展历程（三阶段 + 五模型时间线：RT-1→RT-2→OpenVLA→π0→SmolVLA） |
| 1.4 | 三模态输入架构（ASCII 图：视觉编码器→LLM→动作头） |
| 1.5 | 动作离散化（RT-2 256 桶方案 + 离散化/还原公式） |
| 1.6 | 动作预测头设计（离散分类 / 连续回归 / 扩散动作头） |
| 1.7 | 训练三阶段（预训练→指令微调→动作微调，ASCII 流程图） |
| 1.8 | 连续动作 vs 离散动作（精度/多峰/LLM 融合对比） |
| 1.9 | 实时性要求 + 五大优化手段（量化/蒸馏/推测解码/异步/缓存） |
| 1.10 | 本课程选型理由（OpenVLA → SmolVLA → Pi0 学习路线） |
| 1.11 | VLA 能力边界（长时序/力控/泛化/实时性/安全性） |

### Step 2 — 仿真环境搭建

| 子节 | 内容 |
|------|------|
| 2.1 | 为什么 MuJoCo + LIBERO（四仿真器对比：MuJoCo/Isaac/PyBullet） |
| 2.2 | MuJoCo 核心概念（mjModel/mjData/mj_step + 物理步长 vs 控制周期） |
| 2.2.5 | **动手：加载 6-DOF 机械臂 + 模拟 VLA 控制**（MuJoCo Viewer 实时动画） |
| 2.3 | 环境安装（uv + MuJoCo + LIBERO + GPU 要求） |
| 2.4 | LIBERO 深入（四套件 + gym.Env 接口 + 7 维动作空间详解） |
| 2.5 | 第一个仿真 Demo + 完整实验脚本 |

**配套文件：** `models/widowx_arm.xml` — 6-DOF 臂 + 桌面 + 红色目标方块 + 固定相机

### Step 3 — OpenVLA 实战

| 子节 | 内容 |
|------|------|
| 3.1 | OpenVLA 概述（7B, Stanford+UIUC, 与 RT-2 对比表） |
| 3.2 | 架构详解（SigLIP→Projector→LLaMA→7×256 Action Head，ASCII 数据流图） |
| 3.3 | 模型下载与加载（FP16/INT4/HF 镜像三种方案 + 硬件要求） |
| 3.4 | 推理管线四步骤（图像预处理→Prompt 构建→自回归生成→Token 解码） |
| 3.5 | 完整推理闭环 + run_openvla_libero.py 实验脚本 |
| 3.6 | Token 编码揭秘（32000+dim×256+bucket 映射表） |

### Step 4 — L40 云端训练环境

| 子节 | 内容 |
|------|------|
| 4.1 | 为什么选 L40（四 GPU 对比：L40 44GB / A100 80GB / 3090 24GB / 4060 8GB） |
| 4.2 | SSH 免密登录 + config 别名配置 |
| 4.3 | L40 硬件概览 + PVC 持久化存储说明 |
| 4.4 | uv + PyTorch CUDA 12.x 环境配置 + GPU 验证 |
| 4.5 | scp / rsync 文件传输方案 |
| 4.6 | tmux 后台训练（6 个常用快捷键） |
| 4.7 | GPU 实时监控（nvidia-smi + gpustat） |
| 4.8 | 动手实验：gpu_test.py GPU 基准测试 |

### Step 5 — SmolVLA 轻量 VLA ⭐

| 子节 | 内容 |
|------|------|
| 5.1 | 为什么 SmolVLA（三模型训练对比：参数/显存/成本/时间） |
| 5.2 | 架构对比（EfficientViT vs SigLIP, SmolLM vs LLaMA，逐组件缩减倍数分析） |
| 5.3 | 数据准备（BridgeData V2 + VLADataset 完整实现） |
| 5.4 | 训练管线（~300 行 train.py：AMP + cosine warmup + eval + checkpoint） |
| 5.5 | LIBERO 三套件评估 + SmolVLA vs OpenVLA 结果对比表 |
| 5.6 | 调试改进（5 种常见失败 + 4 个进阶方向） |

**核心结论：** 15x 参数缩减 → 仅 5% 性能下降。SmolVLA 是"最可训的 VLA"。

### Step 6 — Pi0 扩散 VLA ⭐

| 子节 | 内容 |
|------|------|
| 6.1 | 自回归 VLA 三大局限（误差累积 / 离散化精度损失 / 单峰坍缩） |
| 6.2 | 扩散模型直觉（墨水扩散类比 + DDPM vs Flow Matching） |
| 6.3 | Pi0 双系统架构（π0-High 1-5Hz 规划 + π0-Base 100Hz 执行，ASCII 图） |
| 6.4 | Pi0 训练（Flow Matching 目标：预测向量场 + train_pi0.py 核心代码） |
| 6.5 | 自回归 vs 扩散轨迹对比（抖动 0.15 vs 0.05 m/s² / 失败模式分析） |
| 6.6 | 动手实验：Pi0 驱动课程 MuJoCo 机械臂（与 §2 正弦 mock 对比） |

**核心结论：** 扩散 VLA 轨迹更平滑、多峰建模更优，但训练成本是自回归的 3 倍。

### Step 7 — Gemini 真机部署（计划）

| 子节 | 内容 |
|------|------|
| 7.1 | Gemini 硬件回顾（4 臂 + NUC i7 + RTX4060 + 4 路 RGB 相机） |
| 7.2 | 模型转换与部署（L40 checkpoint → ONNX/TensorRT → NUC INT8） |
| 7.3 | 推理管线 + 延迟分析（每段多少 ms） |
| 7.4 | LeRobot 真机适配（运动学参数配置） |
| 7.5 | 安全机制（力限制 / 速度钳制 / 急停 / sim2real gap） |
| 7.6 | 实战：SmolVLA on Gemini 端到端 demo |

### Step 8 — VLA 前沿与竞赛（计划）

| 子节 | 内容 |
|------|------|
| 8.1 | ACoT-VLA：链式思考动作生成 |
| 8.2 | ManipDojo 2026 竞赛回顾 + 课程模型到竞赛的升级路径 |
| 8.3 | 多模态 VLA 趋势（触觉 / 力觉 / 语音） |
| 8.4 | 从 VLA 到具身 AGI——能力边界与未来方向 |

---

## 仿真贯穿线

| 阶段 | 章节 | 仿真内容 | 学生看到什么 |
|------|------|----------|-------------|
| 🔰 入门 | Step 2 | MuJoCo 臂 + 正弦 mock | 机械臂在 Viewer 中摆动 |
| 🤖 推理 | Step 3 | OpenVLA + LIBERO | 大模型根据指令控制机械臂 |
| 🏋️ 训练 | Step 5 | SmolVLA 训练 → LIBERO 评估 | 自己训的模型也能完成任务 |
| 🚀 前沿 | Step 6 | Pi0 训练 → MuJoCo 臂对比 | 扩散轨迹比自回归更平滑 |
| 🎯 落地 | Step 7 | 模型下云 → Gemini 真机 | 仿真里训的模型驱动实体机器人 |

---

## 配套代码

| 文件 | 章节 | 用途 | 验证状态 |
|------|------|------|----------|
| `sim_vla_arm.py` | Step 2 | MuJoCo 机械臂 + mock VLA 控制 | ✅ headless 通过 |
| `run_libero_demo.py` | Step 2 | LIBERO 环境验证 + 随机动作 | ✅ 语法通过 |
| `run_openvla_libero.py` | Step 3 | OpenVLA 7B + LIBERO 零样本推理 | ✅ 语法通过 |
| `gpu_test.py` | Step 4 | L40 GPU 基准测试（TFLOPS+显存） | ✅ 本地通过 |
| `train.py` | Step 5 | SmolVLA 450M 完整训练（含 --dry-run） | ✅ 干跑 200 步通过 |
| `eval_libero.py` | Step 5 | LIBERO 三套件评估 | ✅ 导入验证通过 |
| `sim_pi0_arm.py` | Step 6 | Pi0/Mock 双模式驱动机械臂 | ✅ 语法通过 |
| `widowx_arm.xml` | Step 2+ | 6-DOF 机械臂 MuJoCo 模型 | ✅ 加载通过 |

---

## 环境要求

| 阶段 | 硬件 | 用途 |
|------|------|------|
| Step 2-3 | 任意 Linux + GPU | MuJoCo 仿真 + OpenVLA 推理 |
| Step 4-6 | L40 云端 (44GB) | SmolVLA/Pi0 训练 |
| Step 7 | Gemini 真机 | 实体机器人部署验证 |

---

## 学习路线

```
Step 1 ──→ Step 2 ──→ Step 3 ──→ Step 4 ──→ Step 5 ──→ Step 6 ──→ Step 7 ──→ Step 8
 理论       仿真       推理       算力       训练       前沿       真机       视野
                                    (自回归)   (扩散)
```

**核心能力进阶：** 理解 VLA → 跑通推理 → 训练自己的 VLA → 对比两种范式 → 部署到真机

---

*NXROBO 具身智能课程系列 · License: CC BY-NC-SA 4.0*
