# Step 7: Gemini 大小脑通信

Gemini 小脑 (Jetson Orin NX / NUC) 与 云端大脑 (L40) 之间的 HTTP 通信协议。

## 文件

| 文件 | 角色 | 运行位置 |
|------|------|------|
| `brain_loop.py` | 小脑主循环：采集 → HTTP POST → 执行 | Jetson Orin NX |
| `brain_server.py` | 大脑 API stub：接收观测 → VLA 推理 → 返回动作 | L40 云端 |
| `deploy_model.py` | 旧版：L40→NUC 模型转换 (PT→ONNX→TensorRT) | 已弃用 |
| `configs/record_gemini_3cam.yaml` | 3 相机数据采集配置 | — |
| `grpc/` | 早期 gRPC 版本（SmolVLA），已被 HTTP 版本取代 | — |
| `dm05/` | **DM0.5 (OpenDM) 实机部署**，见下 | NUC + 云端 |

## 快速启动

见 `dm05/README.md`（DM0.5）与 `grpc/README.md`（SmolVLA 旧版）。

## API 协议

`POST /predict` — 观测进、动作出，无状态。

## DM0.5 实机部署（`dm05/`）

用开源的 **DM0.5 (OpenDM)** 基础模型替代自训 SmolVLA，走云端推理 + NUC 执行的
架构。相比自训模型，DM0.5 是 5.83B 预训练 VLA，只需 LoRA 微调（可训练参数
324M / 5.27%）即可适配 LEO-Gemini 的 14 维双臂动作空间。

关键差异与坑（**完整记录见 [`docs/DM05_REALROBOT_DEBUG_NOTES.md`](../../docs/DM05_REALROBOT_DEBUG_NOTES.md)**）：

- **请求必须带 `state`**：`/v1/infer` 无状态，`action_mode=RELATIVE` 时
  服务端做 `state + delta`，传错的 `state` 会直接算出错误的绝对动作。
- **启动必须先整机归位**：训练起始位姿 `right_shoulder ≈ 89.4°`（占 61.44%），
  机器人静止位 `104.7°`（占 0.07%）属分布外，会让模型把右臂驱动得乱走 →
  抓取失败。`--home-mode initial` 会把 14 个关节归位到训练初始位姿。
- **状态读取只有约 3 Hz**（读的是后台线程缓存，不是实时串口轮询），
  归位必须用「分段直发 + 等待到位」，且所有关节**同步**推进而非串行。
- **不要用两跳 SSH 隧道**：延迟尖峰可达 14925 ms，超过 10s 超时阈值会触发
  熔断整轮跳过；改成 NUC 直连云端后最大延迟降到 1306 ms，熔断 0 次。
- **限位单位是 0.1 度**：写成 `(-100, 100)` 会把所有目标夹到 ±10°。
