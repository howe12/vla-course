# DM0.5 (OpenDM) 部署与实机对接 — 技术交接文档

> 生成日期：2026-09-23
> 用途：交给另一个 Agent 在**独立容器**中部署 DM0.5 推理服务，并对接 LEO-Gemini 实体机器人
> 调研来源：GitHub `dexmal/opendm`（已下载源码分析）、官方推理/数据文档、HF 模型卡

---

## 1. 这是什么

**DM0.5** 是原力灵机（Dexmal）开源的**开放世界视觉-语言-动作（VLA）基础模型**，用于通用机器人控制。

| 项 | 值 |
|---|---|
| GitHub | https://github.com/dexmal/opendm |
| 模型权重 | https://huggingface.co/Dexmal/DM05 |
| 技术博客 | https://www.dexmal.com/blog/dm0.5 |
| 许可证 | Apache-2.0（代码）/ Gemma（模型权重） |
| 语言 | Python ≥ 3.10 |

### 官方模型族

| 模型 | 用途 | Chunk | 图片数 | Action 维 |
|---|---|---|---|---|
| **`Dexmal/DM05`** | **基础预训练（本任务用这个）** | **50** | **3** | **14** |
| DM05-libero | LIBERO 仿真 | 10 | 2 | 7 |
| DM05-robotwin2 | RoboTwin 2.0 仿真 | 50 | 3 | 14 |
| DM05-MEM | 带历史帧记忆 | 50 | 3 | 14 |
| DM05-SO101-Pick-Cube | SO101 LoRA 微调 | — | — | — |
| DM05-VLA-Arena | VLA-Arena 仿真 | — | — | — |

**关键匹配点**：基础模型 `3 图 + 14 维 state/action` 的结构，与 LEO-Gemini 双臂
（每臂 6 关节 + 1 夹爪 = 14 维）+ 3 路相机的配置**完全一致**。

---

## 2. 资源需求（评估结论）

| 资源 | 需求 | 备注 |
|---|---|---|
| **GPU** | 推理 **1× GPU 即可**（训练推荐 8×） | L40 46GB 充裕 |
| **显存** | 基础模型 11GB，推理峰值估计 20-24GB | ⚠️ 需实测确认 |
| **磁盘** | 模型 11.1GB + 代码 ~3MB + 依赖 | ⚠️ **磁盘是主要瓶颈** |
| **内存** | 推测 10-20GB（未实测） | 现有容器 128GB 限制，充足 |
| **Python** | ≥3.10（官方建议 conda py3.10） | |

### ⚠️ 磁盘空间警示（重要）

DM0.5 模型 `model.safetensors` = **11.1 GB**，加上依赖约需 **13-18 GB**。
**部署前必须确认目标容器有 ≥20GB 空闲磁盘。**

（参考：当前 gpufree 容器仅剩 7.9GB，不足以部署，需先清理或换容器。）

---

## 3. 依赖清单（来自官方 pyproject.toml）

### 核心依赖（关键版本，勿随意改动）

```
torch==2.11.0                  # 官方指定，从 cu128 索引安装
torchvision==0.26.0
transformers==5.3.0
accelerate==1.14.0
peft==0.19.1
bitsandbytes==0.49.2
tokenizers==0.22.1
numpy==1.26.4                   # 注意是 1.x，不是 2.x
protobuf==7.35.1
pydantic==2.13.4
datasets==5.0.0
diffusers==0.38.0
timm==1.0.27
liger-kernel
einops / einops-exts
fastapi / uvicorn / flask / httpx / requests   # HTTP 推理服务
gradio / wandb / loguru / tyro==1.0.13
decord / av / albumentations                   # 视频/图像处理
scikit-learn / megfile / easydict / shortuuid
```

### 可选：fast 推理后端（**建议先用默认后端，不要装**）

```
[fast-infer]
onnx==1.21.0
triton==3.6.0
tensorrt==10.8.0.43
```

**⚠️ fast 后端是硬依赖而非可选优化**：启用后必须同时满足
`import tensorrt`、`import triton`、`import torch.nn.attention.flex_attention`（torch≥2.5）
三者成功，否则服务**不会自动回退**而是直接失败。**建议第一版用默认 backend。**

### 与现有云端环境的兼容性

现有 gpufree 容器训练环境（SmolVLA）已有 `torch 2.11.0+cu130` / `torchvision 0.26.0`，
**与 OpenDM 要求一致**。但 `transformers`（现有 5.16.1 vs 要求 5.3.0）和
`numpy`（现有 2.2.6 vs 要求 1.26.4）**版本冲突**。

**结论：用独立 conda/venv 环境，不要复用现有训练 env。**

---

## 4. 安装步骤

### 方式 A：Docker（官方推荐，最省心）

```bash
git clone https://github.com/dexmal/opendm.git
cd opendm

docker run -it --rm --gpus all --network host \
  --name opendm \
  --shm-size=16g \
  -v "$PWD":/app/opendm \
  -w /app/opendm \
  dexmal/opendm:latest /bin/bash

# 容器内
conda activate opendm
pip install -e .
```

### 方式 B：本地 conda（推荐用于容器内已有 GPU 透传的场景）

```bash
git clone https://github.com/dexmal/opendm.git
cd opendm

conda create -n opendm python=3.10 -y
conda activate opendm

# 注意：官方指定 cu128 索引
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install ninja packaging
MAX_JOBS=2 pip install flash-attn --no-build-isolation   # 可选，编译慢

pip install -e .
```

### 网络受限环境（国内/容器）的镜像替代

```bash
# pip 走清华镜像
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# torch 官方源不可达时用
# 阿里云 pytorch-wheels: https://mirrors.aliyun.com/pytorch-wheels/cu128
# (需自行确认该目录下有 torch 2.11.0+cu128 的 cp310/cp312 wheel)

# HuggingFace 走镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1      # 关键！hf-mirror 不支持 xet 协议
```

---

## 5. 模型下载

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# 方式1：huggingface-cli
hf download Dexmal/DM05 --local-dir ./checkpoints/DM05

# 方式2：python（更可控，可指定 token）
python -c "
import os; os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
snapshot_download('Dexmal/DM05', local_dir='./checkpoints/DM05')
"
```

### 模型文件清单（HF API 实测）

| 文件 | 大小 |
|---|---|
| `model.safetensors` | **11,118 MB** |
| `tokenizer.json` | 32.6 MB |
| `replay.mp4`（示例视频，可跳过） | 83 MB |
| `config.json` / `norm_stats.json` / tokenizer 配置等 | < 20 KB |

**注意**：checkpoint 目录必须包含与训练配置匹配的 `norm_stats.json`；缺失时
OpenDM 会回退查找 `./norm_stats/` 下的匹配文件。

---

## 6. 启动推理服务

### 基础预训练模型（对应 LEO-Gemini 配置）

```bash
cd opendm

script/dm05_launcher.sh \
  --exp opendm/exp/dm05_exp.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

**关键说明**：
- `--inference-config.image-prompts` 的顺序**决定** HTTP 请求里 `images` 的 `"1"/"2"/"3"` 对应哪路相机
- `robot_type` 建议用 `DOS W1` 或 `Aloha`（两者都是 14 维结构：6关节+夹爪 ×2）
- 直接使用基础模型时**必须显式传** `observation.control_mode` 和 `observation.speed`

### 常用参数表

| 参数 | 说明 |
|---|---|
| `--exp` | playground 入口，须与 checkpoint 匹配 |
| `--model-config.chunk-size` | action horizon，须与训练/客户端一致（基础模型=50） |
| `--inference-config.output-action-dim` | 返回 action 维度（基础模型=14） |
| `--inference-config.image-prompts` | 相机标签，与请求 images 的 "1"/"2"/… 一一对应 |
| `--inference-config.diffusion-steps` | 默认 10 |
| `--inference-config.backend` | `default` 或 `fast` |
| `--inference-config.port` | 默认 7891 |

---

## 7. 连接实体机器人 —— HTTP API 协议

**这是与现有 gRPC 架构最大的差异：DM0.5 用 HTTP + JSON + base64。**

### 请求：`POST /v1/infer`

```bash
curl -X POST http://127.0.0.1:7891/v1/infer \
  -H 'Content-Type: application/json' \
  --data @- <<'EOF'
{
  "observation": {
    "prompt": "grab two objects into the middle box",
    "state": [0.0, 86.8, -85.6, -1.1, -43.4, -5.4, -80.0,
              -2.8, 104.5, -83.7, 0.7, -40.6, -2.1, -79.9],
    "images": {
      "1": "<base64-cam1>",
      "2": "<base64-cam2>",
      "3": "<base64-cam3>"
    },
    "robot_type": "DOS W1",
    "control_mode": "joint",
    "speed": "0.5"
  }
}
EOF
```

### 请求字段详解

| 字段 | 必需 | 说明 |
|---|---|---|
| `observation.prompt` | 否 | 任务指令，默认空字符串 |
| `observation.state` | **是** | 一维 JSON array，**长度和顺序必须与 checkpoint 的 norm_stats 一致**（14 维） |
| `observation.images` | **是** | JSON 对象，值为 base64 图片。**键名必须是连续的 1-based 字符串**（`"1"`,`"2"`,`"3"`），并按顺序对应 `image_prompts` |
| `observation.history_images` | 否 | base64 历史帧数组（旧→新）。**仅在服务以 `--data-config.is-history` 启动时可用** |
| `observation.robot_type` | 否 | 选择归一化 profile，如 `Aloha` / `DOS W1` |
| `observation.control_mode` | 条件必需 | 用基础模型时须显式传（如 `joint` / `eef`） |
| `observation.speed` | 条件必需 | 用基础模型时须显式传，服务默认 `"0.5"` |
| `sampling` | 否 | `num_steps` 须与服务固定 diffusion steps 一致；`seed` 可固定随机性 |

### 响应

```json
{
  "actions": [
    [0.012, -0.034, 0.18, 0.0, 0.0, 0.0, -1.0],
    [0.015, -0.031, 0.17, 0.0, 0.0, 0.0, -1.0]
  ],
  "metadata": { "latency_ms": 123.4 }
}
```

- `actions`：`[chunk_size][action_dim]` 二维数组（基础模型 = `[50][14]`）
- `metadata.latency_ms`：端到端 API 延迟

### ⚠️ 待验证：动作值域语义

响应示例数值在 **-1 ~ 1** 区间，与 state 的 **±100 度**量级差异极大。**必须验证**：

1. DM05 输出是**归一化域**还是**已反归一化**？
2. 若是归一化域 → 需要用 checkpoint 的 `norm_stats.json` 反归一化
3. `action_mode`（relative / absolute）是哪种？relative 模式下夹爪维度保持绝对值

**这是对接实机前必须实测确认的关键点**（参照 SmolVLA 踩坑经验：值域搞错会导致机械臂
被拉向错误位置）。

### Legacy 接口（不推荐新接入使用）

`POST /process_frame` multipart：`text` / `states` / `image`（可重复）/ `robot_type` /
`control_mode` / `speed`。响应格式与 `/v1/infer` 不同（`response` + `model_latency_ms`）。

### 官方示例脚本

```bash
bash tests/curl_demo.sh http://127.0.0.1:7891/v1/infer          # 三图普通请求
bash tests/curl_demo.sh http://127.0.0.1:7891/v1/infer Aloha    # 指定 robot profile
bash tests/curl_history.sh http://127.0.0.1:7891/v1/infer       # 带历史帧（需 is-history 启动）
```

---

## 8. 对接 LEO-Gemini 的适配要点

### 8.1 结构匹配度（很高）

| 项 | DM05 基础模型 | LEO-Gemini | 匹配 |
|---|---|---|---|
| 图片数 | 3 | 3（front/left/right） | ✅ |
| state/action 维 | 14 | 14（双臂 7×2） | ✅ |
| 结构 | 6关节+夹爪 ×2 | 6关节+夹爪 ×2 | ✅ |
| chunk_size | 50 | — | 需客户端适配 |

**LEO-Gemini 的 14 维（waist/shoulder/elbow/forearm_roll/wrist_flex/wrist_roll/gripper）×2
与 DM05 的 `Aloha`/`DOS W1` state_desc 定义完全对应。**

### 8.2 需要做的事

1. **写客户端**：把现有 NUC 的 gRPC client 改成/增加 HTTP client：
   - 采集 3 路相机 → JPEG → base64
   - 读取当前 14 关节位置 → `observation.state`
   - `POST http://<server>:7891/v1/infer`
   - 解析 `actions[50][14]` → 逐步送 `motor_executor`

2. **动作单位对齐**：确认 DM05 输出语义（见 §7 待验证项），与 NUC 的
   `motor_executor`（0.1° 单位 / 度）做正确换算

3. **chunk 执行节奏**：DM05 chunk=50（SmolVLA 是 5），一次推理可支撑更长控制周期，
   但要注意 50 步的执行频率与重规划时机

4. **若需微调**：注册自定义数据集（见 §9），用 LEO 实机数据 SFT

### 8.3 与现有 gRPC 架构的集成方案

```
方案 1（推荐，改动小）：桥接
  NUC gRPC client → 现有 brain_grpc_server → HTTP → DM05 /v1/infer
  （在云端 server 里加一层 HTTP 转发，NUC 侧完全不用改）

方案 2：NUC 直连
  NUC 直接 HTTP 调 DM05（需要网络可达，云端 7891 端口需暴露或走隧道）
  ⚠️ 现有云平台只映射了 SSH 端口，HTTP 需走 SSH 隧道转发：
     ssh -L 7891:localhost:7891 root@<host> -p <port>
```

---

## 9. 微调（如需用 LEO 数据训练）

### 数据格式：JSONL + 图片（**不是 lerobot 格式**）

```
assets/leo_gemini/
├── episode0.jsonl          # 每行一帧
├── episode1.jsonl
├── images/
│   ├── episode0/
│   │   ├── cam_high/0.jpg
│   │   ├── cam_left_wrist/0.jpg
│   │   └── cam_right_wrist/0.jpg
│   └── episode1/
└── index_cache.json        # 自动生成
```

### 单帧 JSON 格式

```json
{"images_1":{"type":"image","url":"./images/episode0/cam_high/0.jpg"},
 "images_2":{"type":"image","url":"./images/episode0/cam_left_wrist/0.jpg"},
 "images_3":{"type":"image","url":"./images/episode0/cam_right_wrist/0.jpg"},
 "state":[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
 "prompt":"grab two objects into the middle box",
 "is_robot":true}
```

**规则**：
- 每个 `.jsonl` = 1 个 episode；每行 1 帧；**不能有空行**；每 episode ≥2 帧
- `state` 每帧必需；`action` 可省略（省略时用未来帧 state 构造目标）
- 若提供 `action`，必须出现在 episode 的**每一帧**
- 修改 jsonl 后要**删除 `index_cache.json`**

### 数据集注册

```python
# opendm/dataset/leo_gemini.py
from opendm.constants.robot import RobotStateDesc, RobotType
from opendm.dataset.register import register_dataset

LEO_STATE_DESC = (
    [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
)

register_dataset({
    "leo_gemini": {
        "jsonl_dir": "./assets/leo_gemini/",
        "image_dir": "./assets/leo_gemini/",
        "image_keys": ["images_1", "images_2", "images_3"],
        "image_prompts": ["Head", "Left wrist", "Right wrist"],
        "robot_type": RobotType.ALOHA,     # 或新增自定义 RobotType
        "state_desc": LEO_STATE_DESC,
        "fps": 30,
    },
})
```

外部注册目录（不改仓库）：
```bash
export OPENDM_DATA_PATH=/abs/path/to/my_dataset_registry
```

### SFT 训练命令

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --task train \
  --nproc_per_node 1 \
  --data-config.dataset-name leo_gemini \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --trainer-config.num-train-steps 5000
```

> ⚠️ 官方推荐训练用 8× GPU；单卡 SFT 可能需 LoRA（参考
> `docs/zh/dm05_so101_lora_training.md`、`docs/zh/dm05_libero_lora_training.md`）。

---

## 10. 已知风险与注意事项

| # | 风险 | 应对 |
|---|---|---|
| 1 | **磁盘不足**：模型 11.1GB + 依赖需 13-18GB | 部署前确认 ≥20GB 空闲 |
| 2 | **fast 后端依赖硬性**且编译/导出耗时长 | 第一版用默认 backend |
| 3 | **动作值域语义未知**（§7 待验证） | 实机前必须离线验证 |
| 4 | **依赖版本冲突**现有 SmolVLA env | 用独立 conda 环境 |
| 5 | **HF 下载 xet 协议不兼容镜像** | 设 `HF_HUB_DISABLE_XET=1` |
| 6 | **norm_stats 缺失导致归一化不匹配** | 确认 checkpoint 带 `norm_stats.json` |
| 7 | **图片顺序错位**：images "1"/"2"/"3" 必须与 image-prompts 顺序一致 | 严格对齐 |
| 8 | 基础模型直接推理可能对 LEO 场景**精度不足**（需微调） | 参照 SmolVLA 经验：环境/背景差异需真实数据微调 |
| 9 | 首个 fast 启动会先导出 ONNX + 建 TRT engine，**服务就绪慢** | 预留启动时间 |
| 10 | HTTP 端口需网络可达 | 云平台需 SSH 隧道转发 7891 |

---

## 11. 与现有 SmolVLA 方案对比

| 维度 | SmolVLA（已跑通） | DM0.5（待部署） |
|---|---|---|
| 模型大小 | 1.2 GB | 11.1 GB |
| 通信 | gRPC（protobuf） | **HTTP + JSON + base64** |
| 端口 | 50051 | 7891 |
| chunk_size | 5 | 50 |
| 输入 | 3 图 + 14 维 state + 文本 | 3 图（base64）+ 14 维 state + prompt |
| 数据格式 | lerobot v3.0（parquet+mp4） | **JSONL + jpg** |
| 训练框架 | lerobot 0.6.1 | OpenDM（自研 trainer） |
| 微调方式 | SFT / resume | SFT / LoRA |
| 推理框架 | PyTorch | PyTorch（可选 TensorRT） |

**结论**：两者可共存于同一 GPU 机器（不同端口/进程），但**依赖环境需隔离**，
且**客户端协议不同**（gRPC vs HTTP），需要新增 HTTP 客户端或做桥接层。

---

## 12. 建议的执行顺序

1. **确认资源**：容器磁盘 ≥20GB、GPU 可见、内存充足
2. **建独立环境**：`conda create -n opendm python=3.10` + `pip install -e .`
3. **下载模型**：`Dexmal/DM05`（11.1GB，走 hf-mirror + 禁用 xet）
4. **启动服务**：default backend，端口 7891，3 图 + 14 维配置
5. **跑通官方 demo**：`bash tests/curl_demo.sh http://127.0.0.1:7891/v1/infer`
6. **验证动作语义**：用 API 返回的 actions 结合 `norm_stats.json` 确认值域和 action_mode
7. **写 LEO HTTP 客户端**：3 相机 base64 + 14 维 state → `/v1/infer` → actions
8. **DRY_RUN 全链路**：不驱动电机，验证动作解析与安全校验
9. **（可选）实机数据微调**：采集 LEO JSONL 数据 → SFT/LoRA
10. **LIVE 验证**：小步长限幅 + 人工监护

---

## 附：原文档索引（仓库内）

| 文档 | 内容 |
|---|---|
| `docs/zh/dm05_inference.md` | **推理服务与 HTTP API（最重要）** |
| `docs/zh/data.md` | 数据格式与数据集注册 |
| `docs/zh/dm05_finetuning.md` | SFT 微调 |
| `docs/zh/robot_platforms.md` | 真机机型改动（AgileX COBOT Magic / Dexmal DOS-W1） |
| `docs/zh/dm05_so101_lora_training.md` | LoRA 微调参考 |
| `docs/zh/dm05_robodojo.md` / `dm05_robotwin2.md` / `dm05_libero.md` | 各 benchmark 流程 |
| `script/dm05_launcher.sh` | 启动脚本（train / inference） |
| `tests/curl_demo.sh` / `curl_history.sh` | HTTP API 调用示例 |
| `third_party/robochallenge_inference/` | 比赛用实机推理客户端参考实现 |
