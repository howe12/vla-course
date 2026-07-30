# 具身大小脑 gRPC 通信模块

> Orin（小脑）↔ 云端 L40（大脑）的 gRPC 双向通信，用于 VLA 模型推理。

## 架构

```
Orin 小脑（client）                    云端 L40 大脑（server）
┌──────────────────────┐              ┌──────────────────────┐
│ camera_capture.py    │              │ brain_grpc_server.py │
│   双路相机采集        │──gRPC──→    │   接收观测            │
│   256×256 JPEG       │   隧道       │   VLA 推理（TODO）    │
│                      │←──gRPC──    │   返回 action chunk   │
│ motor_executor.py    │              │                      │
│   安全限幅+DRY_RUN   │              │ smolvla_libero 模型  │
└──────────────────────┘              └──────────────────────┘
         │
    tunnel_manager.py
    (SSH 隧道 + 健康检查)
```

## 文件说明

| 文件 | 位置 | 说明 |
|---|---|---|
| `embodied_brain.proto` | 两端 | gRPC 接口定义（Observation/ActionChunk/Ping） |
| `embodied_brain_pb2.py` | 两端（生成） | protobuf 消息类 |
| `embodied_brain_pb2_grpc.py` | 两端（生成） | gRPC 服务 stub |
| `brain_grpc_server.py` | 云端 | gRPC server（当前是 stub，TODO 接真实推理） |
| `brain_grpc_client_test.py` | Orin | 延迟测试 client |
| `camera_capture.py` | Orin | 双路相机采集（video0/video2 → 256×256 JPEG） |
| `motor_executor.py` | Orin | 电机执行（DRY_RUN 安全模式，不驱动电机） |
| `tunnel_manager.py` | Orin | SSH 隧道管理（健康检查+指数退避+状态文件） |

## 接口定义（proto）

```protobuf
service BrainService {
  rpc Predict(Observation) returns (ActionChunk);       // 单次推理
  rpc PredictStream(stream Observation) returns (stream ActionChunk);  // 双向流
  rpc Ping(PingRequest) returns (PingResponse);         // 健康检查
}

message Observation {
  repeated Image images = 1;    // 多路 JPEG 图像
  repeated float state = 2;     // 关节状态（8维）
  string instruction = 3;       // 语言指令
  int64 timestamp_ms = 4;       // 时间戳
}

message ActionChunk {
  repeated float actions = 1;   // 展平的动作序列
  int32 chunk_size = 2;         // 步数（50）
  int32 action_dim = 3;         // 维度（7 或 14）
  int64 inference_ms = 4;       // 推理耗时
}
```

## 使用方法

### 1. 启动隧道（Orin，开机自启）

```bash
# 手动管理
~/vla_workspace/scripts/tunnel_ctl.sh status   # 查看状态
~/vla_workspace/scripts/tunnel_ctl.sh test     # 测试连通
~/vla_workspace/scripts/tunnel_ctl.sh restart  # 重启

# systemd 服务（开机自启）
sudo systemctl status grpc-tunnel
```

### 2. 启动云端 server

```bash
# 在云端 L40 上
cd ~/grpc_embodied
~/gpufree-data/vla-course/codes/.venv/bin/python brain_grpc_server.py --port 50051
```

### 3. 运行延迟测试（Orin）

```bash
conda activate lerobot_061
cd ~/grpc_embodied
python brain_grpc_client_test.py
```

### 4. 相机采集测试

```bash
conda activate lerobot_061
cd ~/grpc_embodied
python camera_capture.py
```

### 5. 电机执行测试（DRY_RUN，不驱动电机）

```bash
conda activate lerobot_061
cd ~/grpc_embodied
python motor_executor.py
```

## 延迟实测（2026-07-30）

| 测试 | 平均 | 中位 | 最小 | 最大 |
|---|---|---|---|---|
| Ping RTT | 58.79ms | 50.53ms | 33.74ms | 150.11ms |
| Predict（2路图像） | 100.64ms | 96.89ms | 66.59ms | 157.27ms |

## 安全机制

- **DRY_RUN 模式**：电机执行只记录不驱动（默认）
- **关节限幅**：-100 ~ 100（RANGE_M100_100）
- **单步变化量限制**：max_joint_delta=20（可配置）
- **动作超时**：超过 2000ms 自动跳过
- **隧道健康检查**：每 10s TCP 探测，连续 3 次失败自动重建
- **云端不可用降级**：状态文件 `cloud_available: false`，控制环可据此停止

## 待完成（TODO）

- [ ] 云端 server 接入真实 SmolVLA 推理（需 GPU）
- [ ] 完整控制环组装（相机→gRPC→电机循环）
- [ ] 双向流 PredictStream 实现
- [ ] LIVE 模式电机驱动（需用户授权）
- [ ] 动作维度对齐（smolvla_libero 7维 vs gemini 14/18维）

## 环境依赖

| 端 | 环境 | 关键包 |
|---|---|---|
| Orin | `lerobot_061`（conda） | grpcio 1.83, grpcio-tools, opencv 4.13, torch 2.11 |
| 云端 | `~/gpufree-data/vla-course/codes/.venv` | grpcio 1.83, grpcio-tools, torch 2.10+cu128, lerobot 0.4.4 |

## 网络拓扑

```
Orin (192.168.100.56)
  └── SSH 隧道 (systemd: grpc-tunnel)
        └── localhost:50051 → 120.209.70.195:30334 → 云端容器:50051
```

- Orin 主动拨出（NAT 后无法被云端直连）
- 隧道由 `tunnel_manager.py` 管理（指数退避重连 + 状态文件）
- 云平台只映射了 SSH 端口 30334，gRPC 通过 SSH 隧道复用
