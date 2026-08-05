# 🤝 LEO-Gemini 具身 VLA 项目交接文档

> 生成日期：2026-08-05
> 本文档是**项目全貌索引**，任何 Agent 接手时先读本文件，再按需深入各部分。

---

## 0. 一句话项目概述

Jetson Orin NX (LEO-Gemini 双臂机器人) 上采集数据，微调 VLA 模型（SmolVLA/ACT/Pi0.5），云端 L40 训练推理 + Orin 实机执行（大小脑 gRPC 架构）。

---

## 1. 设备与环境

### 设备
| 项 | 值 |
|---|---|
| 主机 | Jetson Orin NX 16GB，192.168.100.56 |
| 登录 | `nxrobo@192.168.100.56`（密码 `nxrobo`） |
| 系统 | Ubuntu 22.04.5 / JetPack 6.2 / L4T R36.5.0 |
| GPU | CUDA 12.6 / cuDNN 9.3 / TensorRT 10.3 |

### 软件环境（conda）
| 环境 | Python | torch | lerobot | 关键包 | 用途 |
|---|---|---|---|---|---|
| `lerobot`（旧） | 3.10 | 2.7.0 | 0.2.0 | — | 保留 |
| `lerobot_061`（主） | 3.10 | 2.11.0 cu126 | **0.6.1** | grpcio/opencv GUI/av 15.1/pyserial | 采集/推理/训练 |

激活：`source ~/miniconda3/etc/profile.d/conda.sh && conda activate lerobot_061`

### 服务（systemd，开机自启）
| 服务 | 功能 |
|---|---|
| `x11vnc` | VNC 远程桌面（5900，密码 nxrobo，自动登录 :0） |
| `grpc-tunnel` | SSH 隧道连云端 L40（localhost:50051→云端） |

---

## 2. 硬件（当前状态）

### 相机（4 个，USB）
| 键名 | video | by-path | USB | 状态 |
|---|---|---|---|---|
| top | video0 | usb-0:1.1 | USB3 | ✅ |
| right | video6 | usb-0:2.4 | USB2 | ✅ |
| left | video2 | usb-0:2.2.2 | USB2 | ✅ |
| front | video4 | usb-0:2.3 | USB2 | ✅（可换 right） |

> ⚠️ **带宽限制**：USB2 总线最多 2 个 Microdia 相机同时采集。稳定方案：top + 任意 2 个 Microdia。
> 详见 `docs/USB_CAMERA_BANDWIDTH_CONCLUSION.md`

### 机械臂（LEO-Gemini 双臂，14 关节）
| 串口 | 用途 |
|---|---|
| /dev/ttyACM_left_follower | 左臂执行 |
| /dev/ttyACM_left_leader | 左臂主手 |
| /dev/ttyACM_right_follower | 右臂执行 |
| /dev/ttyACM_right_leader | 右臂主手 |
| /dev/ttyACM_lift | 升降 |

> ⚠️ **当前机械臂电机（HDSC 2e88:4603）未上电**——上次检测只有升降（f618:0620）在线。采集前需确认电机上电。

---

## 3. 关键配置与脚本

### 采集配置（主）
```
~/vla-course/codes/step7_gemini/configs/record_gemini_3cam.yaml
```
- 相机：top + right + left（by-path 稳定绑定）
- 统一 640×480 @ 25fps
- dataset.root: ~/.cache/huggingface/lerobot

### 脚本清单
```
~/vla-course/workspace/scripts/
├── camera_live_4x.py        # 相机实时预览（--mode 3a/3b/4）
├── camera_capture.py        # 相机采集模块（by-path 绑定）
├── camera_preview.py        # 相机可视化（旧版）
├── test_teleop_camera.py    # 遥操作+相机测试（安全模式）
├── read_motor_state.py      # 电机只读状态读取
├── motor_executor.py        # 电机执行（DRY_RUN 安全）
├── tunnel_ctl.sh            # SSH 隧道管理
├── act_gpu_smoke.py         # ACT GPU 测试
├── act_latency.py           # 延迟基准
└── gpu_proof.py             # GPU 验证
```

---

## 4. 大小脑架构（gRPC）

```
Orin 小脑 (client)                 云端 L40 大脑 (server)
  相机采集 + 电机执行  ←gRPC→   SmolVLA 推理
  (经 SSH 隧道 50051)
```
- proto: `codes/step7_gemini/grpc/embodied_brain.proto`
- 实测延迟：Ping ~50ms，Predict ~100ms
- 云端 L40：root@120.209.70.195:30334（密码 rma6qr9j）

---

## 5. 数据集（本地 ~/.cache/huggingface/lerobot，约 55G）

| 数据集 | episode | 相机 | 任务 |
|---|---|---|---|
| sandiqwq/ok/gemini_test_tea | 50 | top+front | 放杯+倒水+盖盖 |
| sandiqwq/ok/gemini_test_towel | 50 | top+front | 捡毛巾放置 |
| sandiqwq/ok/gemini_test_opengai | 50 | top+front | 开瓶盖 |
| sandiqwq/ok/gemini_test_grasp_obj_fifty | 50 | top+front | 抓取入盒 |
| sandiqwq/grasp2box | 100 | top+front | 红方入蓝盒 |
| sandiqwq/move2box | 100 | top+front | 底盘导航 |
| sandiqwq/new/gemini_test_grasp_obj | 20 | **top+front+left** | 传感器+开关入盒 |

- 已上传 HF：6 个数据集到 HoweXixi12（公开）
- HF 账号：sandiqwq（本地）/ HoweXixi12（上传）

---

## 6. 当前状态与待办

### ✅ 已完成
- 环境搭建（lerobot_061 + torch 2.11 cu126）
- 相机绑定（by-path 稳定）+ 带宽问题诊断
- 采集配置固化（25fps 统一）
- 大小脑 gRPC 通信链路
- 电机只读读取验证

### ⏳ 待办（下一步）
1. **机械臂电机上电** → 验证串口（4 个 HDSC）
2. **遥操作测试** → `test_teleop_camera.py --drive` 小范围试动
3. **正式采集** → `lerobot-record --config_path ...` 采集任务数据
4. **云端训练** → L40 上用 smolvla_libero 微调
5. **部署推理** → gRPC server 接真实 SmolVLA

### ⚠️ 已知问题
- 机械臂电机未上电（HDSC 不在 USB 枚举）
- USB2 带宽限制 4 相机（软件无法解决，需 USB3 相机）
- 云端 lerobot 0.4.4 vs Orin 0.6.1 版本差异（训练产物需验证兼容）

---

## 7. 常用命令速查

```bash
# 连接 Orin
ssh nxrobo@192.168.100.56

# 激活环境
conda activate lerobot_061

# 相机预览
python3 ~/vla-course/workspace/scripts/camera_live_4x.py --mode 3a

# 采集（top+right+left, 25fps）
lerobot-record \
  --config_path ~/vla-course/codes/step7_gemini/configs/record_gemini_3cam.yaml \
  --dataset.repo_id "HoweXixi12/任务名" \
  --dataset.single_task "任务指令"

# 采集控制：→ 提前结束当前段 / ← 重录上一段 / Esc 停止

# 电机只读
python3 ~/vla-course/workspace/scripts/read_motor_state.py

# VNC
地址 192.168.100.56:5900，密码 nxrobo
```

---

## 8. 云端 L40

- SSH: `root@120.209.70.195:30334`（密码 `rma6qr9j`）
- 环境: `~/gpufree-data/vla-course/codes/.venv`（Python 3.10, torch 2.10+cu128, lerobot 0.4.4）
- 模型: `~/gpufree-data/models/`（smolvla_libero, pi05_libero, openvla-7b）
- 仿真: IsaacLab / IsaacSim / LIBERO
- ⚠️ 无卡模式（需云平台切换有卡实例才可推理）
