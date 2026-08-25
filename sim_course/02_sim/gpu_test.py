#!/usr/bin/env python
"""L40 GPU 基准测试 - 【需要 GPU 开启后运行】
启动前请先确认: nvidia-smi 能看到 L40, 或 torch.cuda.is_available() 为 True。
用途: 验证 L40 算力基线 (TFLOPS + 显存), 见飞书 02-2。
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, time

def banner():
    print("=" * 50)
    print("L40 GPU 基准测试")
    print("=" * 50)

if not torch.cuda.is_available():
    print("❌ CUDA 不可用，GPU 未挂载或未开启。请在平台侧开启 GPU 后重试。")
    raise SystemExit(1)

dev = torch.device("cuda:0")
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
print(f"GPU: {name}")
print(f"显存: {props.total_memory/1e9:.1f} GB")
print(f"SM 数 / 多处理器: {props.multi_processor_count}")

# (TFLOPS) FP32 矩阵乘粗测
a = torch.randn(8192, 8192, device=dev)
b = torch.randn(8192, 8192, device=dev)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
el = (time.perf_counter() - t0) / 5
flops = 2 * 8192**3
print(f"FP32 矩阵乘: {flops/el/1e12:.1f} TFLOPS (理论峰值约 60+, 实测为粗值)")

# 显存占用
import torch as _t
buffer = _t.zeros(8 * 1024, 1024, 1024, dtype=torch.float16, device=dev)
print(f"分配 8GB 显存后: 已用 {torch.cuda.memory_allocated()/1e9:.1f} GB")
del buffer
torch.cuda.empty_cache()
print("=== GPU 基线测试完成 ===")