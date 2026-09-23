# SmolVLA 实机控制部署手册 — 2026-09-21 工作记录

> **目的**：记录 SmolVLA 模型从云端推理到 NUC 机械臂控制的完整链路，包括归一化处理、单位转换、安全机制、踩坑与解决方案。避免未来重复摸索。
>
> **涉及设备**：
> - 云端 L40（120.209.70.195:30215）— SmolVLA 推理服务
> - NUC（192.168.100.146）— 相机采集 + gRPC client + 电机执行
> - LEO-Gemini 双臂机器人 — SgrMotorsBus 串口控制

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  云端 L40 (brain_grpc_server_real.py)                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SmolVLAPolicy.from_pretrained(checkpoint_002000)    │    │
│  │   ↓ predict_action_chunk(batch)                     │    │
│  │   ↓ raw_actions * std + mean  ← 反归一化           │    │
│  │   ↓ 输出: 度 (degree)                              │    │
│  └─────────────────────────────────────────────────────┘    │
│         ↑ gRPC Predict(Observation)                         │
│         ↓ gRPC ActionChunk(actions=[5][14], 单位=度)        │
├─────────────────────────────────────────────────────────────┤
│  SSH 隧道 (paramiko, NUC localhost:50051 → 云端:50051)      │
├─────────────────────────────────────────────────────────────┤
│  NUC (motor_executor_live.py)                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. 采集 3 路相机 (front/left/right) → JPEG base64   │    │
│  │ 2. 读取当前 14 关节位置 (度) → ×10 → 0.1°          │    │
│  │ 3. gRPC Predict(obs) → actions[5][14] (度)         │    │
│  │ 4. actions × 10 → 0.1° 单位                        │    │
│  │ 5. MotorExecutor 安全校验 (限幅+变化量截断)          │    │
│  │ 6. live.send_actions(0.1°) → lerobot send_action   │    │
│  │    ↓ val_0p1 × 0.1 → 度                            │    │
│  │    ↓ lerobot SgrMotorsBus: int(val_deg × 10)       │    │
│  │    ↓ 写入电机 Goal_Position (int16, 0.1° 单位)      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 单位转换全链路（最关键）

### 2.1 各单位定义

| 位置 | 单位 | 说明 |
|---|---|---|
| 训练数据集 action/state | **度 (degree)** | 如 shoulder=86.8 表示 86.8° |
| 模型内部（归一化空间） | 无量纲 | raw ≈ 0 附近，通过 mean/std 映射 |
| checkpoint postprocessor | mean/std | `action.mean` / `action.std`（度单位） |
| brain_grpc_server 输出 | **度 (degree)** | `raw * std + mean` 反归一化后 |
| gRPC ActionChunk.actions | **度 (degree)** | 传输层单位 |
| NUC motor_executor_live 接收 | 度 → **×10 → 0.1°** | `MODEL_TO_DEG_INV = 10.0` |
| MotorExecutor.last_positions | **0.1°** | 安全校验的内部单位 |
| live.send_actions 输入 | **0.1°** | executor 传出 |
| live.send_actions 内部 | 0.1° → **×0.1 → 度** | `MODEL_TO_DEG = 0.1` |
| lerobot GeminiFollower.send_action | **度** | `{name.pos: degree}` |
| SgrMotorsBus.write("Goal_Position") | **int16 (0.1°)** | `int(degree × 10)` |
| 电机硬件 | **0.1°** | 协议原生单位 |

### 2.2 转换公式汇总

```
云端推理:
  real_degree = raw_normalized × action_std + action_mean

NUC 接收:
  value_0p1deg = action_degree × 10          # MODEL_TO_DEG_INV

executor 安全校验后:
  safe_0p1deg → live.send_actions(safe_0p1deg)

send_actions 内部:
  degree = safe_0p1deg × 0.1                  # MODEL_TO_DEG
  robot.send_action({f"{name}.pos": degree})

lerobot → 电机:
  raw_int16 = int(degree × 10)                # SgrMotorsBus 内部
```

### 2.3 ⚠️ 踩坑记录：单位搞错的后果

| 错误 | 现象 | 修复 |
|---|---|---|
| 把模型输出当 0.1° 直接用 | 机械臂被拉向 ±10° 附近（实际应在 ±86°） | 确认模型输出是"度"，需 ×10 转 0.1° |
| state 填 0.1° 值（如 868）但模型期望度（86.8） | 模型"看不懂"当前姿态，输出回到均值 | state 用度单位填入 |
| 未做初始位置对齐 | 首步 delta 巨大（如 868→28 = 840），被截断到 max_delta | executor.last_positions 初始化为当前真实姿态 |

---

## 3. 反归一化实现细节

### 3.1 stats 来源

从 checkpoint 的 `policy_postprocessor_step_0_unnormalizer_processor.safetensors` 读取：

```python
import safetensors.torch
state = safetensors.torch.load_file(f"{model_path}/{state_file}")
action_mean = state["action.mean"].float()   # shape [14]
action_std = state["action.std"].float()     # shape [14]
```

### 3.2 实测验证（离线回放）

用训练数据集中的真实帧喂给模型，对比输出 vs 真实 next action：

| frame | state shoulder | 模型输出 | 真实 next | MAE |
|---|---|---|---|---|
| 500 | -9.8° | [-1.2, -0.2] | [-14.1, 7.2] | 6.4 |
| 1000 | 84.8° | [77.2, -76.7] | [88.7, -88.9] | 10.2 |
| 5000 | 84.4° | [70.2, -73.0] | [88.7, -88.8] | 8.4 |

**结论**：模型学到了大致方向，但精度 ~10-18° MAE。在训练图像上能输出合理动作，换到 NUC 不同背景时退化到均值附近。

### 3.3 attention_mask 必须是 bool

```python
# ❌ 错误：Long tensor 导致 RuntimeError
mask = enc["attention_mask"].to(device)

# ✅ 正确：显式转 bool
mask = enc["attention_mask"].to(device).bool()
```

---

## 4. 安全机制详解

### 4.1 MotorExecutor 三层防护

```
模型输出 (度)
    ↓ ×10
0.1° 单位
    ↓
① 关节限幅: clamp(-100, 100)  ← RANGE_M100_100
    ↓
② 单步变化量截断: |target - last| > max_joint_delta → 截断
    ↓
③ 超时熔断: elapsed > action_timeout_ms → 跳过该步
    ↓
safe_targets → live.send_actions()
```

### 4.2 首次执行超时 bug 及修复

**问题**：`execute_action_chunk` 开头检查 `elapsed > timeout`，但首次调用时 `last_action_time` 是初始化时刻（可能已过数秒），导致首步就被判超时跳过，且**超时路径不更新 last_action_time** → 后续每步都超时 → 全部跳过。

**修复**（已应用到 NUC 上的 motor_executor.py）：
```python
# 修改前
elapsed_ms = (now - self.last_action_time) * 1000
if elapsed_ms > self.config.action_timeout_ms:
    logger.warning(...)
    return

# 修改后
if self.action_count > 0:  # ← 首次执行跳过超时检查
    elapsed_ms = (now - self.last_action_time) * 1000
    if elapsed_ms > self.config.action_timeout_ms:
        logger.warning(...)
        return
```

### 4.3 LIVE 模式注入机制

**不修改 motor_executor.py 核心逻辑**，通过注入回调实现：

```python
executor.live_driver = live.send_actions  # 注入
# executor._execute_live() 内部调用 self.live_driver(targets)
```

motor_executor.py 的 `_execute_live` 改为：
```python
def _execute_live(self, targets):
    if self.live_driver is None:
        raise RuntimeError("LIVE 驱动回调未注入")
    self.live_driver(targets)
```

---

## 5. 初始位置对齐

### 5.1 为什么需要对齐

模型输出的目标是绝对位置（度）。如果 executor 的 `last_positions` 从 0 开始，而实际机械臂在 shoulder=86.8°，则：
- 模型目标 28° → delta = |28 - 0| = 28（0.1° 单位下是 280）
- 被 max_joint_delta=10 截断 → 每步只动 1°
- 需要 58 步才能到达目标，效率极低

### 5.2 对齐方法

```python
# LIVE 模式下
init_state = live.read_state_0p1deg()  # 读当前真实姿态 (0.1°)
executor.last_positions = dict(zip(JOINT_NAMES, init_state))
```

这样首步 delta = |280 - 868| = 588 → 截断到 10 → 从当前位置渐进移动。

### 5.3 read_state_0p1deg 实现

```python
def read_state_0p1deg(self):
    obs = self.robot.get_observation()
    pos = {}
    for k, v in obs.items():
        if k.endswith(".pos"):
            name = k.removesuffix(".pos")
            pos[name] = round(v * 10, 1)  # lerobot 返回度 → ×10 → 0.1°
    return [pos.get(name, 0.0) for name in JOINT_NAMES]
```

**注意**：连接后需 `time.sleep(1.5)` 等待电机就绪，否则首次读取可能返回全 0。

---

## 6. 相机配置

### 6.1 NUC 相机映射

| video 节点 | 名称 | by-path | 类型 |
|---|---|---|---|
| /dev/video0 | front | usb-0:1 | Realtek USB Camera3 |
| /dev/video2 | left | usb-0:2.2 | Microdia Integrated_Webcam_HD |
| /dev/video5 | right | usb-0:3 | Microdia Integrated_Webcam_HD |

```python
CAMERA_MAP = [(0, "front"), (2, "left"), (5, "right")]
```

### 6.2 采集参数

- 分辨率：640×480
- JPEG 质量：85
- 预热：每路先读 2 帧丢弃（避免首帧黑帧/过曝）

### 6.3 相机绑定脚本

`camera_bind.py` 提供交互式绑定工具，支持：
- 自动枚举相机
- 挥手检测识别物理位置
- 生成 udev 规则持久化

---

## 7. 云端 Server 启动

```bash
# 在云端容器内
cd /root/gpufree-data/vla-course/codes/step7_gemini/grpc/
export HF_ENDPOINT=https://hf-mirror.com

python brain_grpc_server_real.py --port 50051 \
  --model /root/gpufree-data/robotics-vla/checkpoints/smolvla_grasp_two_obj/checkpoints/002000/pretrained_model
```

加载时间约 28s（模型 450M 参数 + tokenizer）。

---

## 8. NUC 侧运行

### 8.1 DRY_RUN 测试（不驱动电机）

```bash
conda activate lerobot
cd ~/vla-grpc/codes/step7_gemini/grpc/
python motor_executor_live.py --rounds 3 --max-joint-delta 10 --step-delay 0.3
```

### 8.2 LIVE 模式（真实驱动）

```bash
echo "ENABLE" | python motor_executor_live.py --live --rounds 20 --max-joint-delta 10 --step-delay 0.3
```

**参数说明**：
- `--max-joint-delta 10`：每步最多 1°（10 × 0.1°），保守安全
- `--step-delay 0.3`：步间间隔，给电机响应时间
- `--rounds 20`：20 轮 × 5 步 = 100 步执行

---

## 9. SSH 隧道

NUC 通过 paramiko 建立 SSH 隧道到云端：

```python
# tunnel_test.py（已在 NUC 上运行）
ssh.connect("120.209.70.195", port=30215, username="root", password="...")
transport = ssh.get_transport()
listener.bind(("127.0.0.1", 50051))
# accept → open_channel("direct-tcpip", ("127.0.0.1", 50051), addr)
```

隧道进程 PID 可通过 `ps aux | grep tunnel_test` 查看。

---

## 10. 已知限制与后续改进

| # | 问题 | 现状 | 改进方向 |
|---|---|---|---|
| 1 | 模型精度不足（MAE 10-18°） | 2000 步训练太少 | 增加训练步数或用 NUC 数据微调 |
| 2 | 背景差异导致输出退化 | NUC 背景 ≠ 训练背景 | 采集 NUC 环境数据重训 |
| 3 | 模型不执行抓取动作 | 只回归到训练分布中心 | 需任务特定微调 |
| 4 | chunk_size=5 较小 | 每轮只执行 5 步 | 可增大或改用 DM0.5（chunk=50） |
| 5 | 推理延迟 ~280ms | L40 默认 backend | TensorRT fast backend 可优化 |
| 6 | 相机视角未验证 | NUC 相机安装位置可能与训练时不同 | 需人工确认画面内容 |

---

## 11. 文件清单

| 文件 | 位置 | 用途 |
|---|---|---|
| `brain_grpc_server_real.py` | 云端 grpc/ | 真实 SmolVLA 推理 gRPC server |
| `motor_executor_live.py` | NUC grpc/ | LIVE/DRY_RUN 控制客户端 |
| `motor_executor.py` | NUC grpc/ | 安全执行器（已 patch 首次超时 bug） |
| `camera_bind.py` | NUC grpc/ | 相机绑定工具 |
| `dry_run_loop.py` | NUC grpc/ | DRY_RUN 全链路测试 |
| `check_unnorm2.py` | 本地 | 反归一化验证脚本 |
| `verify_align.py` | 本地 | 初始位置对齐验证 |
| `verify_real_cam_nuc.py` | 本地 | 真实相机+state 推理验证 |
| `tunnel_test.py` | NUC /tmp/ | SSH 隧道（paramiko） |
| `embodied_brain.proto` | 两端 grpc/ | gRPC 接口定义 |
| `embodied_brain_pb2*.py` | 两端 grpc/ | proto 编译产物 |

---

## 12. 快速恢复指南

如果需要在新的 NUC/云端环境重新部署：

1. **云端**：scp `brain_grpc_server_real.py` + proto 文件 → 启动 server
2. **NUC**：确保 conda lerobot env 有 grpcio + opencv + lerobot
3. **NUC**：scp `motor_executor_live.py` + `motor_executor.py`（含 patch）
4. **NUC**：建立 SSH 隧道（tunnel_test.py 或手动 ssh -L）
5. **NUC**：先跑 `--rounds 3` DRY_RUN 验证链路
6. **NUC**：确认安全后 `--live` 驱动

**不需要重新训练模型、重新编译 proto、重新下载 checkpoint**——这些都在云端持久化了。
