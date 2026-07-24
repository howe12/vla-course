# Step 7: Gemini 大小脑通信

Gemini 小脑 (Jetson Orin NX) 与 云端大脑 (L40) 之间的 HTTP 通信协议。

## 文件

| 文件 | 角色 | 运行位置 |
|------|------|------|
|  | 小脑主循环：采集 → HTTP POST → 执行 | Jetson Orin NX |
|  | 大脑 API stub：接收观测 → VLA 推理 → 返回动作 | L40 云端 |
|  | 旧版：L40→NUC 模型转换 (PT→ONNX→TensorRT) | 已弃用 |

## 快速启动



## API 协议


