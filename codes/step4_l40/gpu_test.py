#!/usr/bin/env python3
"""Step 4 实验：L40 GPU 基准测试

验证 GPU 环境可用：CUDA、显存、算力。

用法（在 L40 上）：
    uv run python codes/step4_l40/gpu_test.py
"""

import sys

try:
    import torch
except ImportError:
    print("❌ PyTorch 未安装")
    print("请运行: uv pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

import time


def main():
    print("=" * 50)
    print("Step 4 实验：L40 GPU 基准测试")
    print("=" * 50)

    # ============================================================
    # 1. 基本信息
    # ============================================================
    print(f"\n[1/3] 环境信息")
    print(f"  PyTorch: {torch.__version__}")

    if not torch.cuda.is_available():
        print("  ❌ CUDA 不可用！请检查 PyTorch 是否安装了 CUDA 版本")
        print("     运行: uv pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3

    print(f"  GPU:     {device_name}")
    print(f"  VRAM:    {vram_gb:.1f} GB")
    print(f"  Compute: {props.major}.{props.minor}")
    print(f"  SM:      {props.multi_processor_count}")

    if vram_gb < 16:
        print(f"  ⚠️  显存只有 {vram_gb:.1f} GB，SmolVLA 训练可能 OOM（需要 16-24GB）")
    elif vram_gb < 30:
        print(f"  ✅ 显存 {vram_gb:.1f} GB，可以训练 SmolVLA")
    else:
        print(f"  ✅ 显存 {vram_gb:.1f} GB，可以训练 SmolVLA 和 Pi0")

    # ============================================================
    # 2. 算力测试
    # ============================================================
    print("\n[2/3] 矩阵乘法算力测试 (FP16)")
    for size in [1024, 2048, 4096, 8192]:
        try:
            a = torch.randn(size, size, device="cuda", dtype=torch.float16)
            b = torch.randn(size, size, device="cuda", dtype=torch.float16)

            # 预热
            for _ in range(5):
                torch.matmul(a, b)
            torch.cuda.synchronize()

            # 计时
            start = time.time()
            for _ in range(20):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            tflops = 2 * size**3 * 20 / elapsed / 1e12
            ms_per_op = elapsed / 20 * 1000
            print(f"  {size:5d}×{size:<5d}  {ms_per_op:6.1f}ms  ({tflops:.1f} TFLOPS)")

            del a, b
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  {size:5d}×{size:<5d}  ❌ OOM")
            break

    # ============================================================
    # 3. 显存测试
    # ============================================================
    print("\n[3/3] 显存分配测试")
    for frac in [0.25, 0.5, 0.75, 0.9]:
        try:
            total = props.total_memory
            n_floats = int(total * frac) // 4
            x = torch.zeros(n_floats, device="cuda")
            used = torch.cuda.memory_allocated() / 1024**3
            print(f"  分配 {frac*100:3.0f}% 显存: {used:.1f} GB  ✅")
            del x
            torch.cuda.empty_cache()
        except RuntimeError:
            print(f"  分配 {frac*100:3.0f}% 显存: ❌ OOM")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 50)
    print("✅ GPU 基准测试完成！")

    if vram_gb >= 16:
        print(f"  环境可用于 Step 5 SmolVLA 训练 ({vram_gb:.0f}GB VRAM)")
    if vram_gb >= 30:
        print(f"  环境可用于 Step 6 Pi0 训练 ({vram_gb:.0f}GB VRAM)")

    print(f"  ⏭️  下一步: Step 5 — 训练 SmolVLA 450M 模型")
    print("=" * 50)


if __name__ == "__main__":
    main()
