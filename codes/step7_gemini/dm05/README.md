# DM0.5 实机部署（LEO-Gemini）

用 **DM0.5 (OpenDM)** 基础模型驱动 LEO-Gemini 双臂机器人：云端 GPU 跑推理，
NUC 负责采图、读关节状态、安全校验与电机下发。

**完整调试记录（含根因分析与实测数据）见
[`docs/DM05_REALROBOT_DEBUG_NOTES.md`](../../../docs/DM05_REALROBOT_DEBUG_NOTES.md)。**

## 架构

```
NUC ──3×USB相机(video0/4/2) + 14关节状态──► HTTP POST /v1/infer ──► 云端 L40
 ◄────────────── 50 步 × 14 维动作序列 ────────────────────────────┘
 │
 └─ 限位 / 限速 / 超时熔断 → 逐步下发 GeminiFollower（双臂串口）
```

`/v1/infer` **无状态**，每轮必须带上当前 `state`；`action_mode=RELATIVE`（默认）时
服务端做 `state + delta`，所以 **`state` 直接参与最终动作计算**。

## 文件

| 文件 | 作用 |
|---|---|
| `motor_executor.py` | 安全执行器：限位、限速、超时熔断、首次动作跳过 |
| `motor_executor_live_dm05.py` | 主控：读状态/图像、调用推理、整机归位、逐步下发 |
| `brain_http_client.py` | HTTP 客户端：3 路图像编码、front 裁剪、缺失相机补灰图 |
| `dry_run_dm05.py` | 干跑（不驱动电机）验证链路 |
| `probe_home.py` | 独立归位探针：只归位、不推理，验证归位逻辑 |
| `tunnel_dm05.sh` | SSH 隧道（端点走环境变量，不硬编码） |

## 环境准备

```bash
# NUC
source /opt/conda/etc/profile.d/conda.sh   # 或使用自带 venv
# 需要: numpy opencv-python pyserial lerobot

# 云端：DM0.5 在无 flash-attn 环境需打 3 个补丁
# 见 docs/DM05_CLOUD_PATCH_BACKUP.md
```

## 用法

```bash
# 1) 建立隧道（NUC 直连云端，不要用两跳转发）
export DM05_REMOTE_HOST=<推理服务主机>
export DM05_REMOTE_PORT=<ssh 端口>
./tunnel_dm05.sh &

# 2) 先只验证归位（安全）
python3 probe_home.py 5.0
#   输出「归位前 / 目标 / 归位后 / 残余」对照表，14 关节应在 ±1.5° 内

# 3) 干跑，确认链路
python3 dry_run_dm05.py

# 4) 实机运行（默认先整机归位到训练初始位姿）
python3 motor_executor_live_dm05.py \
    --live --home-mode initial --rounds 30 \
    --max-joint-delta 5 --step-delay 0.05 \
    --max-delta0-warn 20 --flush-camera
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--live` | 关 | 真实驱动电机；不加则为 DRY_RUN |
| `--home-mode` | `initial` | `initial`=整机归位到训练初始位姿；`grip`=仅开夹爪；`off`=不归位 |
| `--rounds` | 5 | 控制回合数 |
| `--max-joint-delta` | 20 | 单步最大变化量（**0.1 度**单位，20 = 2°/步） |
| `--step-delay` | 0.05 | 每步间隔（秒） |
| `--max-delta0-warn` | 15 | `chunk[0]` 与 state 偏差超此值（度）则跳过本轮 |
| `--addr` | `http://127.0.0.1:7891` | 推理服务地址 |
| `--loose-limits` | 关 | 换用更宽的机械边界（默认用训练范围 `SAFE_LIMITS`） |
| `--flush-camera` | 关 | 开相机后排空缓冲，取真正最新帧 |
| `--grip-open` | 20.0 | 启动时夹爪张开角度（度） |

## 安全注意

- ISO/TS 15066 协作机器人末端限速 **0.25 m/s**。`--max-joint-delta 5`
  ≈ 0.12 m/s（调试推荐），`15` ≈ 0.43 m/s 已超限，**不建议**。
- 调试期务必用 `--max-joint-delta 5`，确认稳定后再逐级放宽。
- `--home-mode initial` 会驱动**全部 14 个关节**，启动前确认机械臂活动范围内
  无障碍物、急停可达。
- 关闭扭矩后**夹爪会机械弹回 −80°**（满闭），右肩回到约 98–104°。
  归位必须在同一个已连接并已使能扭矩的进程内完成，不能跨进程。

## 相机槽位映射（2026-09-28 定稿）

**这是最容易踩的坑：物理归属 ≠ 训练配对。**

- 物理上（单爪张开对照实验验证）：`video2` = 左臂相机、`video4` = 右臂相机
- 但数据集采集时两路腕部标签与物理臂是**交叉**的，模型学到的是交叉配对
- 推理端必须复现交叉 ⇒ **`left→video4(USB 3-3)`、`right→video2(USB 3-2.2)`**
  （代码 `CAMERA_USB_PATH` 固化；`--camera-map` 可临时覆盖）
- ⚠️ USB 端口编号会漂移（今天 3-2.2 就从 video2 变成了 video3），所以按**路径**绑定

确认方法（换线后必做）：
```bash
# 归位后单独张开一侧夹爪，看哪路腕部画面剧变（自视特征）
python3 probe_which_cam2.py
# 行为判据: 正确配对 → 夹爪张开到 ~50 再闭合到 ~10；错误 → 夹爪几乎不动
```

## 新鲜帧（2026-09-28 新增，防 USB 饥饿静态帧）

`video2`（挂在与两条随动板共用的 3-2 集线器上）在电机运动时会**反复返回
逐字节相同的陈旧缓冲帧**（实测 23/30 轮），导致模型"睁眼瞎"地重复记忆动作。
`motor_executor_live_dm05.py` 已内置 `_grab_fresh_frame`：帧时间戳 + 内容
双重确认，取不到新帧就重抓（1s 超时兜底+告警）。修复后实测 0/15 轮静态帧。

## 训练（v3 重训，详细见 `docs/DM05_V3_TRAINING_REPORT.md`）

- 单卡必须 `--nproc_per_node 1`（默认 8 会 OOM）
- bs4 × accum2 = 有效批 8（与旧一致，LR 语义不变），比旧 bs1×accum8 快 45%
- `save_only_model=True` ⇒ 无优化器状态，**不可中断续训**
- 推荐 ckpt：`leo_lora_v3/checkpoint-1500`（0.46 epoch，离线回放全面胜出）
