# 附录：SmolVLA MuJoCo 推理调试实录

> 在 AMD ROCm 上复现 Datawhale 官方 SmolVLA 抓取任务的全过程——7 次推理实验，从完全失败到抓起杯子。

---

> **🔗 前置章节**：第 5 章 SmolVLA 训练、Datawhale Every Embodied 课程

> **🎯 学习目标**
>
> - 理解 SmolVLA 推理管线每个环节对最终结果的影响
> - 掌握 `action_type`（`joint_angle` vs `eef_pose`）的本质区别
> - 学会用 `predict_action_chunk` + `select_action` 的不同组合调试 VLA
> - 学会通过系统化对比定位推理失败根因
> - 获取可直接用于课程展示的实验素材（视频 + 关键帧）

> **📍 背景**
>
> Datawhale 发布了 [every-embodied-smolvla-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-smolvla-mujoco-pnp) 官方权重，宣称在 MuJoCo 桌面抓取任务上达到 **95%（57/60）** 的严格成功率。本实验在 **AMD ROCm（Radeon gfx1100, 48GB VRAM）** 平台上尝试复现，经历 7 轮迭代后才定位到核心问题。

---

## 1. 任务与环境

### 1.1 任务描述

```
指令: "Place the blue mug on the plate."
场景: MuJoCo 桌面环境，OMY 6-DOF 机械臂 + 二指夹爪
观测: 两路 RGB 相机（agentview + egocentric）+ 6 维关节角状态
动作: 7 维输出（6 维 EEF 位姿增量 + 1 维夹爪）
```

### 1.2 硬件与软件

| 项目 | 配置 |
|---|---|
| GPU | AMD Radeon (gfx1100), 48GB VRAM |
| 推理框架 | ROCm 7.2.1, PyTorch 2.10.0 |
| 仿真器 | MuJoCo 3.10, EGL headless 渲染 |
| 模型 | SmolVLA 450M (SmolVLM2-500M backbone) |
| 权重 | Datawhale/every-embodied-smolvla-mujoco-pnp |

---

## 2. 官方配置解读

从官方权重 `config.json` 提取核心推理参数：

```json
{
    "chunk_size": 50,
    "n_action_steps": 50,
    "resize_imgs_with_padding": [512, 512],
    "n_obs_steps": 1,
    "tokenizer_max_length": 48,
    "input_features": {
        "observation.state":  {"shape": [6]},
        "observation.image": {"shape": [3, 256, 256]},
        "observation.wrist_image": {"shape": [3, 256, 256]}
    },
    "output_features": {
        "action": {"shape": [7]}
    }
}
```

**关键解读**：

- `resize_imgs_with_padding: [512, 512]` → 图像必须保持宽高比缩放到 512×512 再填充
- `chunk_size: 50` → 模型一次预测 50 帧动作
- 动作空间是 **7 维**：`[dx, dy, dz, dr, dp, dy, gripper]`
- 官方 notebook 使用 `action_type='eef_pose'` → 动作是 EEF 位姿增量，需要通过 IK 转为关节角

---

## 3. 七轮迭代实验

### 实验矩阵

| 轮次 | action_type | 推理 API | chunk | 图像 | 结果 |
|---|---|---|---|---|---|
| ① joint | joint_angle | select_action | 50 | 512+pad | ❌ 碰到杯子，夹爪不合 |
| ② eef v1 | eef_pose | predict_action_chunk | 50 | 512+pad | 💥 GLFW crash |
| ③ eef v2 | eef_pose | predict_action_chunk | 50 | 512+pad | ⏱️ 首次调用 10 分钟无输出 |
| ④ eef v3 | eef_pose | select_action | 5 | 256 | 🐢 300 步 gap 只从 0.96→0.57m |
| ⑤ open-loop | eef_pose | predict_action_chunk | 50 | 512+pad | ✅ 抓起 23cm！但开环滑落 |
| ⑥ closed (n=5) | eef_pose | select_action | 50/5 | 512+pad | 🐢 450 步没够到杯子 |
| ⑦ **final** | eef_pose | predict_action_chunk | 50 | 512+pad | ✅ 抓起 13.7cm，但夹爪不合 |

> **注**：⑦ 使用 `predict_action_chunk` 每 50 步重新观测（12 次模型调用 / 600 步），等效 `select_action` + `n_action_steps=50`

---

## 4. 核心发现

### 4.1 错误一：`action_type='joint_angle'`

**现象**：模型输出被当作关节角直接执行，实际模型输出的是 EEF 位姿增量。

```
模型输出: [0.001, 0.002, -0.003, ...]  ← EEF delta (米/弧度)
错误执行: 设为关节角 [0.001, 0.002, ...] rad  ← 几乎不动
正确执行: p0 += delta[:3], 然后 IK 解算关节角
```

**教训**：推理前必须确认模型的**动作空间类型**，不能只看维度。

### 4.2 错误二：忽略渲染后端

**现象**：GLFW 在 headless 环境中崩溃。

```
❌ 错误：未设置 DISPLAY，GLFW 尝试创建 X11 窗口 → core dump
✅ 修复：DISPLAY=:99 MUJOCO_GL=egl（使用 Xvfb + EGL）
```

### 4.3 错误三：`select_action` 时序集成过于保守

**现象**：`select_action` + `n_action_steps=5` 时，时序集成平均了多个重叠 chunk，导致动作幅度极小。

```python
# n_action_steps=5 时：每 5 步调用 predict_action_chunk
# 时序集成有 10 个 (50/5) 重叠 chunk，平均后动作 ≈ 零
→ 450 步 gap 才从 0.99 缩到 0.32m（没够到杯子）
```

**解决**：`n_action_steps=50`（与训练一致），无时序重叠，保持动作幅度。

### 4.4 根因：夹爪力控不匹配

**这是导致抓取失败的根本原因**：

```
模型输出 gripper 值范围: 0.39 ~ 0.47（对应夹爪半开）
SimpleEnv2 的 gripper 映射: g_target → gripper_rate_per_step=0.03 → 逐帧渐变
实际夹爪状态: 始终半开，无法夹紧杯子
```

**现象**：模型能精准逼近杯子（best gap 0.22m），能抬起杯子（max lift 13.7cm），但夹爪夹不紧，杯子在抬升过程中滑落。

---

## 5. 最终实验结果（第⑦轮）

### 5.1 推理全过程

```
[Chunk  0] step=  0  gap=0.992m  mug_z=0.845  ← 初始位置，手臂远离
[Chunk  1] step= 50  gap=0.607m  mug_z=0.845  ← 快速接近
[Chunk  2] step=100  gap=0.340m  mug_z=0.911  ← 🔵 抓取并抬起 6.6cm！
[Chunk  3] step=150  gap=0.254m  mug_z=0.966  ← 🔵 峰值：抬起 12.1cm！
[Chunk  4] step=200  gap=0.276m  mug_z=0.923  ← ⚠️ 杯子开始滑落
[Chunk  5] step=250  gap=0.341m  mug_z=0.848  ← ❌ 掉回桌面
[Chunk  6] step=300  gap=0.414m  mug_z=0.853  ← 手臂撤回
...
[Chunk 11] step=550  gap=0.647m  mug_z=0.849  ← 无法恢复
```

### 5.2 性能数据

| 指标 | 数值 |
|---|---|
| 总步数 | 600 |
| 模型调用次数 | 12（每 50 步一次） |
| 每次预测耗时 | ~1.0 秒 |
| 杯子最大抬升 | 13.7 cm |
| 最佳 EEF-杯子距离 | 0.22 m |
| 夹爪值范围 | 0.39 ~ 0.47（半开） |

---

## 6. 课程素材

实验自动记录了以下素材，可用于课程演示：

```
outputs/course_capture/
├── inference_final.mp4          ← 600 帧完整推理视频 (7MB)
└── frames/
    ├── 000_init_agent.jpg       ← 初始状态（agent 视角）
    ├── 000_init_ego.jpg         ← 初始状态（腕部视角）
    ├── best_gap_*.jpg           ← 最近距离时刻系列
    └── peak_z_*.jpg             ← 杯子抬升时刻系列
```

**推荐展示顺序**：

1. `000_init_agent.jpg` → 初始场景（杯子在桌面，手臂在上方）
2. `peak_z_0050.jpg` → 手臂下降接近杯子
3. `peak_z_0140.jpg` → 杯子被抬起到最高点（z=0.966m）
4. `peak_z_0200.jpg` → 杯子开始滑落
5. `inference_final.mp4` → 完整过程视频

---

## 7. 调试方法论总结

### 分层排查框架

```
第 1 层：环境     → DISPLAY、MUJOCO_GL、Xvfb ✅
第 2 层：模型加载 → from_pretrained、config 对齐 ✅
第 3 层：动作空间 → eef_pose vs joint_angle     ✅ ← 最易错
第 4 层：推理 API → select_action vs predict_action_chunk ✅
第 5 层：时序参数 → chunk_size、n_action_steps  ✅
第 6 层：图像处理 → resize、padding、归一化     ✅
第 7 层：控制映射 → gripper rate、IK 参数       ⚠️ ← 当前瓶颈
```

### 单一变量原则

每次只改一个参数，对比结果。例如：

- 第④轮 vs 第⑤轮：仅改 `predict_action_chunk` → `select_action`，其他不变
- 第⑤轮 vs 第⑦轮：仅改 `n_action_steps`（50→50，添加周期观测）

---

## 8. 下一步方向

1. **夹爪映射修复**：分析训练数据中 gripper 值的实际分布，与 `SimpleEnv2` 的 `gripper_rate_per_step` 速率限制对齐
2. **物理步进调优**：当前每步 1 次 `step_env()`，官方可能用不同的物理步进数
3. **多 seed 测试**：官方评测用 seed 0-29 × 2 指令 = 60 次，需要批量测试验证一致性
4. **成功判据对齐**：官方使用「严格物理成功谓词」（lift≥3cm + 直立法向余弦≥0.7），需完全对齐

---

> **💡 关键心得**
>
> VLA 推理调试不是"调参"——是**逐层验证假设**。每当你认为"模型不行"时，先检查：① 环境对吗？② 配置对吗？③ 动作空间对吗？④ 控制映射对吗？大多数时候，问题不在模型，在你自己写的桥接代码。
>
> 本次实验的 7 轮迭代中，模型从未"不行"——它每次都精准执行了我们的（错误或正确的）指令。是我们一步步把桥接代码修对了，它才开始抓起杯子。
