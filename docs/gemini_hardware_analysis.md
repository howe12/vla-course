# Gemini 硬件系统分析

> SSH: nxrobo@192.168.100.56:22 | ROS2 Humble 2.2.3 | 采集时间 2026-07-23

## 1. 系统信息

| 项目 | 规格 |
|------|------|
| OS | Ubuntu 22.04, Linux 6.8.0-71-generic x86_64 |
| CPU | Intel i7-14650HX, 20 核 28 线程 (P-core 2.2GHz + E-core 1.6GHz) |
| 内存 | 31 GiB total, ~20 GiB used (ROS + 相机驱动占用大) |
| GPU | NVIDIA RTX 4060 8GB (nvidia-smi 不可用 — 驱动未加载或容器环境) |
| 网络 | enp8s0: 192.168.100.56 |
| ROS2 | Humble Hawksbill 2.2.3, Fast DDS |
| 工作空间 | `~/humble_ws`, `~/ros2_ws`, `~/spark_ws` |

## 2. 机器人控制节点 (`nologo_joy_ctrl`)

运行中的核心控制节点，发布以下 Topic：

### 关节指令
| Topic | 类型 | 说明 |
|-------|------|------|
| `/joint_command` | Float64MultiArray | **VLA 输出的目标关节角** |
| `/target_joint_pos` | 同类型 | 目标位置 |
| `/mode` | Int32 | 模式切换 (主从/自主/急停) |

### 传感器反馈
| Topic | 类型 | 说明 |
|-------|------|------|
| `/joint_states` | JointState | 当前关节角 + 角速度 (ROS 标准) |
| `/motor_states` | 自定义 | 电机状态 (电流/温度/错误码) |
| `/force_data` | 自定义 | 关节力矩反馈 |
| `/real_force_data` | 自定义 | 真实力传感器数据 |
| `/real_pos_force_data` | 自定义 | 位置 + 力混合数据 |
| `/real_pos` | Float64MultiArray | 实际关节位置 |
| `/arm_status` | 自定义 | 臂状态 (到位/运行中/错误) |
| `/robot_pose_status` | 自定义 | 机器人末端位姿 |
| `/rpy_status` | Float64MultiArray | Roll/Pitch/Yaw 状态 |

### 相机
| Topic | 类型 | 频率 |
|-------|------|------|
| `/left_cam/image_raw` | Image | 30Hz |
| `/left_cam/compressed` | CompressedImage | 30Hz |
| `/left_cam/camera_info` | CameraInfo | — |
| `/right_cam/image_raw` | Image | 30Hz |
| `/right_cam/compressed` | CompressedImage | 30Hz |
| `/right_cam/camera_info` | CameraInfo | — |

## 3. 关键架构观察

1. **双相机（左/右）** — 第7章说"4路相机"但实际运行的是左右两路
2. **关节指令通过 `/joint_command`** — VLA 输出直接写入此 topic 即可控制
3. **力传感器活跃** — 安全监控（L2 力保护）可直接读取 `/real_force_data`
4. **模式切换 `/mode`** — 可切换控制模式（遥操作 vs 自主推理）
5. **GPU 不可用** — RTX 4060 驱动未加载，可能需要在 NUC 本机上调试

## 4. 大小脑架构适配

```
Gemini (小脑)                        云端 L40 (大脑)
┌──────────────────────┐          ┌─────────────────────┐
│ nologo_joy_ctrl 控制  │          │ VLA 推理容器          │
│                      │  相机图  │                     │
│ left_cam ────────────┼──压缩──→│ 接收图像+关节状态     │
│ right_cam ───────────┤  像      │                     │
│ joint_states ────────┼──关节角→│ 推理 → 动作           │
│                      │          │                     │
│      ←── 动作 ───────┼──/joint │ SmolVLA / Pi0 / ACT │
│                      │  _command│                     │
│ IK + 安全 执行        │          │                     │
└──────────────────────┘          └─────────────────────┘
```

带宽估算: 2×640×480 压缩图 ≈ 200KB/frame × 15Hz = 3MB/s (局域网可承受)
