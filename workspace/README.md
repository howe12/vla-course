# VLA 工作区 — Jetson Orin NX

> 本目录是 VLA（视觉-语言-动作）项目在 Jetson Orin NX 上的工作记录归档，供后续 Agent 接手分析。
> 读完本 README 即可了解设备环境、已验证能力、数据集与复现方法。

---

## 1. 设备画像

| 项 | 值 |
|---|---|
| 型号 | Jetson Orin NX 16GB（Engineering Reference Dev Kit） |
| SSH | `nxrobo@192.168.100.56`（密码 `nxrobo`） |
| 系统 | Ubuntu 22.04.5 LTS / JetPack 6.2 / L4T R36.5.0 |
| CPU | 8 核 ARM Cortex-A78AE @ 1.5GHz（当前 25W 模式，BogoMIPS 62.5） |
| 内存 | 16GB 统一内存 + 7.6GB zram swap |
| GPU | Orin 集成 GPU（SoC，CPU/GPU 共享内存） |
| ML 栈 | CUDA 12.6 / cuDNN 9.3 / TensorRT 10.3 / Docker 29.6 |
| 磁盘 | 233G，当前使用率约 43%（已清理编译残留） |

**注意**：Jetson 是 SoC 集成 GPU，`nvidia-smi` 的 memory/utilization 显示 `N/A` 是正常现象；看 GPU 负载请用 `jtop`（jetson-stats）或 `tegrastats`。

---

## 2. Conda 环境（双环境并存）

| 环境 | Python | torch | lerobot | CUDA | 用途 |
|---|---|---|---|---|---|
| `lerobot`（旧，保留） | 3.10 | 2.7.0 cu126 | 0.2.0 | ✅ | 原有环境，未改动 |
| `lerobot_061`（新） | 3.10 | 2.11.0 cu126 | 0.6.1（editable） | ✅ | 新建，已验证 |

激活新环境：
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_061
```
（`lerobot_061` 已通过 conda `activate.d` 钩子自动设置 `LD_LIBRARY_PATH`，含 `nvidia/cu12/lib`，无需手动 export。）

`lerobot_061` 关键包：torch 2.11.0、torchvision 0.26.0、transformers 5.5.4、accelerate 1.14.0、num2words 0.5.14、nvidia-cudss-cu12 0.8.0.10。

---

## 3. 已验证能力（GPU 全链路）

用 **ACTPolicy**（纯 PyTorch ResNet18 骨干，无需 VLM 权重）验证了 lerobot 0.6.1 在 Jetson GPU 上的完整链路：

- 模型实例化：51.6M 参数，全部参数 + buffer 在 `cuda:0`
- 训练前向（VAE）：loss 有限（l1_loss + kld_loss）
- 推理 `select_action`：输出张量 `device == cuda:0`
- **GPU 显存占用 220.8MB**

**性能基线**（25W 模式，batch=1，480×640）：
- 单次推理延迟：**72.7 ms**
- 吞吐：**13.8 actions/s**（满足实时控制 ≥10Hz）

**判断 GPU 是否被正确使用的黄金判据**：不看 `torch.cuda.is_available()`，而看模型 `.to(cuda)` 后最终输出张量的 `.device` 是否为 `cuda:0`。

---

## 4. 数据集清单（`~/.cache/huggingface/lerobot/`，共约 55G，全 30fps）

### 高质量抓取类（gemini_follower 双臂）
| 数据集 | ep | frames | 大小 | 任务 |
|---|---|---|---|---|
| `sandiqwq/ok/gemini_test_tea` | 50 | 75100 | 1.9G | 拿物体入杯+倒水+盖盖（复杂多步） |
| `sandiqwq/ok/gemini_test_towel` | 50 | 62595 | 1.3G | 捡毛巾放置 |
| `sandiqwq/ok/gemini_test_opengai` | 50 | 37599 | 855M | 捡水瓶开盖 |
| `sandiqwq/ok/gemini_test_grasp_obj_fifty` | 50 | 25098 | 606M | 捡物体入盒 |
| `sandiqwq/new/gemini_test_grasp_obj` | 20 | 26511 | 1.1G | 传感器入白盒+开关入透明盒 |

### 移动底盘类（sgr_mobile_base_serial）
| 数据集 | ep | frames | 大小 | 任务 |
|---|---|---|---|---|
| `sandiqwq/grasp2box` | 100 | 30991 | 380M | 捡红方块入蓝盒 |
| `sandiqwq/move2box` | 100 | 39747 | 419M | 左转前进到蓝盒（导航） |

### 中小规模 / 测试性质
| 数据集 | ep | frames | 大小 | 备注 |
|---|---|---|---|---|
| `sandiqwq/yes/gemini_test_grasp_obj_two` | 30 | 15059 | 373M | 捡物体入盒 |
| `sandiqwq/new/gemini_test_grasp_obj_one` | 20 | 14226 | 352M | 捡桌上物体放右盒 |
| `sandiqwq/yes/gemini_test_grasp_obj_one` | 20 | 10040 | 248M | 捡物体入盒 |
| `sandiqwq/base/base-only-train` | 30 | 22553 | 157M | 底盘旋转（mobile_base_serial） |
| `mytest/gemini_test_188` | 6 | 3557 | 99M | 测试，规模小 |
| `mytest/new_test_01_20260710_175523` | 1 | 648 | 12M | 测试，无 task，质量低 |

数据集格式：`data/chunk-000/*.parquet` + `meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}` + `videos/chunk-000/`。

---

## 5. 验证脚本

位于本目录 `scripts/`（原始副本也在 `~/`）：
- `act_gpu_smoke.py` — ACT GPU 全链路冒烟测试（实例化+forward+select_action）
- `act_latency.py` — ACT 推理延迟基准
- `gpu_proof.py` — GPU 使用证明（逐参数 device 检查 + 显存占用）
- `smolvla_gpu_smoke.py` — SmolVLA 冒烟测试（备用，需 VLM 权重）

运行方式：
```bash
conda activate lerobot_061
python ~/vla-course/workspace/scripts/act_gpu_smoke.py
```

---

## 6. 关键踩坑点（重要！）

1. **torch wheel 来源**：Jetson 不能用 PyPI 的 torch（那是 x86/桌面 cu130 wheel，CUDA 不可用）。必须用 jetson-ai-lab 预编译索引：
   `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/`（cu126 只有 cp310，无 cp312 → 故选 Python 3.10）。
2. **避免源码编译**：从源码编译 torch 需 5-6 小时（6571 编译单元，ARM CPU 慢），且易因网络/OOM 失败。预编译 wheel 是正解。
3. **libcudss.so.0 缺失**：JetPack 6.2 未含，`pip install --no-deps nvidia-cudss-cu12` + LD_LIBRARY_PATH 指向 `site-packages/nvidia/cu12/lib`。**切勿全量装**（会拉 cuda-toolkit 12.9 与系统 CUDA 12.6 冲突）。
4. **torchcodec 陷阱**：在 aarch64 要求 torch>=2.11，会把 pip 拽到桌面版 cu130。纯推理不需要 torchcodec，安装 lerobot 时排除。
5. **代理必需**：Jetson 上网（下载 wheel、访问 HF Hub）必须走代理 `http://192.168.100.122:7897`（直连仅 16KB/s）。HF 操作示例：
   `env https_proxy=http://192.168.100.122:7897 http_proxy=http://192.168.100.122:7897 hf auth whoami`
6. **lerobot 0.6.1 在 Python 3.10 的 12 处 patch**（已直接改在 `~/source_new/lerobot/` 源码）：
   - `pyproject.toml`：`requires-python >=3.12 → >=3.10`，删 torchcodec/torch/torchvision 依赖行
   - `configs/video.py`：`typing.Self → typing_extensions.Self`
   - `utils/io_utils.py`：PEP695 `def func[T]()` → TypeVar
   - `processor/pipeline.py`：`class X[T1,T2](HubMixin)` → `(HubMixin, Generic[T1,T2])`
   - `datasets/streaming_dataset.py`：`class Backtrackable[T]` → `(Generic[T])`
   - `motors/motors_bus.py`：`type X = Y` → `Union[X,Y]`
   - 7 个 `modeling_*.py`：`typing.Unpack → typing_extensions.Unpack`

---

## 7. 遗留 / 后续

- **SmolVLA 权重缺失**：`~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/` 只有 tokenizer，无模型权重。需从 HF 下载 `lerobot/smolvla_base` 或用自训 checkpoint。
- **MAXN 提速**（可选）：`sudo nvpmodel -m 0 && sudo jetson_clocks` 可提升推理速度（需 sudo，改变功耗）。
- **FP16/TensorRT 加速**（可选）：进一步降低延迟。
- **数据集上传 HF Hub**：HF 已登录账号 `sandiqwq`（走代理可用），待确认上传哪些数据集。

---

## 8. 维护信息

- 创建时间：2026-07-29
- HF 账号：`sandiqwq`
- 控制端记忆文件：`jetson-orin-vla-env.md`（Reasonix 项目记忆）

---

## 9. 具身大小脑 gRPC 通信模块（~/vla-course/codes/step7_gemini/grpc/）

Orin（小脑）↔ 云端 L40（大脑）的 gRPC 双向通信，用于 VLA 模型推理。

### 文件说明

| 文件 | 说明 |
|---|---|
| `embodied_brain.proto` | gRPC 接口定义（Observation/ActionChunk/Ping） |
| `embodied_brain_pb2.py` / `_grpc.py` | 生成的 protobuf/gRPC stub |
| `brain_grpc_server.py` | 云端 gRPC server（当前 stub，TODO 接真实推理） |
| `brain_grpc_client_test.py` | Orin 端延迟测试 client |
| `camera_capture.py` | 双路相机采集（video0/video2 → 256×256 JPEG） |
| `motor_executor.py` | 电机执行（DRY_RUN 安全模式，不驱动电机） |
| `tunnel_manager.py` | SSH 隧道管理（健康检查+指数退避+状态文件） |
| `README.md` | 详细文档（架构图/接口/使用方法/延迟数据） |

### 隧道管理

```bash
~/vla-course/workspace/scripts/tunnel_ctl.sh status   # 查看隧道状态
~/vla-course/workspace/scripts/tunnel_ctl.sh test     # 测试连通性
~/vla-course/workspace/scripts/tunnel_ctl.sh restart  # 重启隧道
```

### 延迟实测（2026-07-30）

- Ping RTT: 平均 58.79ms / 中位 50.53ms
- Predict（2路图像）: 平均 100.64ms / 中位 96.89ms

### 安全机制

- DRY_RUN 模式：电机只记录不驱动（默认）
- 关节限幅 + 单步变化量限制 + 超时检测
- 隧道健康检查 + 云端不可用降级

详见 ~/vla-course/codes/step7_gemini/grpc/README.md`。
