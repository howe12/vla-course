# LEO-Gemini 相机 USB 带宽排查结论

> 日期：2026-08-05
> 设备：Jetson Orin NX (NXROBO LEO-Gemini)
> 问题：4 路相机无法同时采集

## 一、结论速览

| 项 | 结论 |
|---|---|
| **能否 4 相机同时采集** | ❌ **不能**（USB2 带宽硬限制） |
| **根本原因** | 3 个 Microdia 相机共享 USB2 总线（bus1）的 480Mbps，isoc 带宽预留超载 |
| **修改分辨率能否解决** | ❌ **不能**（640×480 已是相机硬件下限） |
| **MJPG 能否解决** | ❌ **不能**（isoc 按最大预留，不按实际数据量） |
| **挪到 USB3 口能否解决** | ❌ **不能**（Microdia 是 USB2 设备，无法使用 SuperSpeed） |
| **当前可用相机数** | ✅ **3 相机**（top + 任意 2 个 Microdia） |
| **数据集实际需求** | ✅ **3 相机**（top + front + left），已满足 |

## 二、硬件拓扑（实测）

```
Orin NX (tegra-xusb 控制器)
├── bus1 (USB2, 480Mbps)  ← 3 个 Microdia 相机 + 串口 + 蓝牙 共享
│   ├── Type-C 口 (1-1) → 外接 Hub → 机械臂串口
│   └── 板载 USB2 Hub (1-2)
│       ├── Port3 → front 相机 (Microdia, 640x480)
│       ├── Port4 → right 相机 (Microdia, 640x480)
│       └── Port2 → 二级 Hub → left 相机 (Microdia, 640x480)
│
└── bus2 (USB3, 5Gbps)  ← top 相机独享（不拥堵）
    └── USB3 Hub (2-1) → top 相机 (USB Camera3, 5Gbps)
```

关键事实：
- **bus1 和 bus2 是两条独立总线**，带宽互不影响
- **3 个 Microdia 都在 bus1**（Type-C 口和板载 Hub 共享同一 root hub）
- **top 在 bus2**（USB3），独享 5Gbps，不是瓶颈

## 三、排查过程与证据

### 1. 相机支持的分辨率（v4l2-ctl 枚举）
```
Microdia 相机：
  MJPG: 1280x720@30fps, 640x480@25fps
  YUYV: 1280x720@10fps, 640x480@25fps
→ 640x480 已是最小分辨率，无法再降
```

### 2. 实测组合（OpenCV 采集）
| 组合 | 结果 |
|---|---|
| top + front + left | ✅ 全部 OK（20/20 帧） |
| top + right + left | ✅ 全部 OK（20/20 帧） |
| top + front + right | ✅ 全部 OK |
| front + left + right（无 top） | ❌ 第 3 个 Microdia 失败 |
| 4 相机全开 | ❌ 第 3 个 Microdia 失败 |
| 换打开顺序（right 先开） | ❌ 仍有一个 Microdia 失败 |

**规律**：无论顺序，**3 个 Microdia 必有一个失败** → bus1 最多支持 2 个 Microdia。

### 3. 内核日志（决定性证据）
```
usb 1-2.4: Not enough bandwidth for new device state.
usb 1-2.4: Not enough bandwidth for altsetting 3
```
`altsetting 3` 是 UVC 相机的视频流配置，内核无法为其分配足够的 isoc（等时）带宽。

### 4. USB 设备能力确认
```
Microdia: bcdUSB 2.00（USB 2.0 设备）
USB Camera3 (top): USB 3.0（5Gbps）
```

## 四、为什么这些方案都无效（原理）

### 1. 降分辨率无效
```
USB isoc 带宽预留 = 分辨率 × 像素字节 × fps
相机硬件最小分辨率 = 640x480
→ 3 个相机的"最小预留"仍超过 bus1 容量
→ 分辨率不是可调变量
```

### 2. MJPG 无效
```
关键：USB isoc 带宽按"altsetting 声明的最大包大小"预留
     不按"实际传输的数据量"预留

MJPG 让实际数据变少，但 UVC 驱动仍按该格式的
最大传输能力预留 isoc 带宽 → 预留没减少
```

### 3. 挪到 USB3 口无效
```
Microdia 是 USB 2.0 设备（bcdUSB 2.00）
→ 即使插到 USB3 口，仍按 USB2 协商
→ isoc 带宽仍走 bus1 (USB2 控制器)
→ 不占用 bus2 的 5Gbps SuperSpeed
```

### 4. 带宽计算
```
bus1 (USB2): 480Mbps，isoc 可用约 384Mbps（80%）
每相机 640x480 的 isoc 预留 ≈ 147Mbps（YUYV）或略低（MJPG）
3 相机累加 > 384Mbps → 第 3 个必然失败
```

## 五、解决方案

### 当前可用（推荐）
```
✅ 3 相机方案：top + 任意 2 个 Microdia
   → 已实测稳定（20/20 帧）
   → 正好满足数据集需求（top + front + left）
```

### 若需 4 相机（硬件方案）
| 方案 | 说明 | 可行性 |
|---|---|---|
| 换 USB3 相机 | USB3 相机走 bus2（5Gbps），不占 bus1 | ✅ 唯一根治 |
| PCIe USB 扩展卡 | 独立 USB 控制器 = 独立带宽 | ⚠️ 需确认 PCIe 口 |
| 接受 3 相机 | top + 2 个 Microdia | ✅ 立即可用 |

## 六、配置固化（采集时免长参数）

配置文件：`codes/step7_gemini/configs/record_gemini_3cam.yaml`

```bash
lerobot-record \
  --config_path ~/vla-course/codes/step7_gemini/configs/record_gemini_3cam.yaml \
  --dataset.repo_id "sandiqwq/数据集名" \
  --dataset.single_task "任务指令"
```

## 七、附：相关资源

- 相机可视化脚本：`workspace/scripts/camera_live_4x.py`（--mode 3a/3b/4）
- 采集配置：`codes/step7_gemini/configs/record_gemini_3cam.yaml`
- 相机绑定（by-path）：`workspace/scripts/camera_capture.py`
