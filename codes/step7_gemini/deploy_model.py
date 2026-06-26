#!/usr/bin/env python3
"""Step 7 实验：VLA 模型转换 — PyTorch → ONNX → TensorRT INT8

在 L40 上运行，将训练好的 SmolVLA/Pi0 checkpoint 导出为 TensorRT 引擎，
供 Gemini NUC (RTX 4060 8GB) 高效推理。

用法（L40 上）：
    uv run python codes/step7_gemini/deploy_model.py --ckpt outputs/smolvla/best.pt
    uv run python codes/step7_gemini/deploy_model.py --ckpt outputs/pi0/best.pt --model pi0
"""

import argparse
import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("❌ PyTorch 未安装")
    sys.exit(1)


# 导入课程模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step5_smolvla"))
from train import SmolVLA


def export_onnx(model, output_path, image_size=224, device="cuda"):
    """PyTorch 模型 → ONNX"""
    print(f"\n[1/3] 导出 ONNX: {output_path}")

    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    # 尝试用实际推理验证一次
    with torch.no_grad():
        test_out = model(dummy_input)
    action_dim = test_out.shape[-1]
    print(f"  模型输出维度: {action_dim}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["action"],
        dynamic_axes={"image": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"  ✅ ONNX 导出完成: {output_path} ({size_mb:.1f} MB)")
    return output_path


def verify_onnx(onnx_path, image_size=224):
    """验证 ONNX 模型可以加载和推理"""
    print("\n[2/3] 验证 ONNX 模型...")

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  ⚠️  onnx/onnxruntime 未安装，跳过验证")
        print("  安装: pip install onnx onnxruntime")
        return

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"  ✅ ONNX 模型结构验证通过")

    # 推理测试
    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    out = session.run(None, {"image": dummy})

    print(f"  ✅ ONNX 推理测试通过")
    print(f"  输入: (1, 3, {image_size}, {image_size})")
    print(f"  输出: {out[0].shape}")


def print_tensorrt_instructions(onnx_path, model_name):
    """打印 TensorRT 转换命令（需在 NUC 上执行）"""
    engine_path = onnx_path.replace(".onnx", "_int8.engine")

    print(f"\n[3/3] TensorRT 转换指令（在 Gemini NUC 上执行）")
    print("=" * 60)
    print(f"# 1. 复制 ONNX 到 NUC")
    print(f"scp {onnx_path} gemini@<nuc-ip>:/home/gemini/models/")
    print()
    print(f"# 2. 在 NUC 上转换 TensorRT + INT8")
    print(f"trtexec \\")
    print(f"    --onnx=/home/gemini/models/{os.path.basename(onnx_path)} \\")
    print(f"    --saveEngine=/home/gemini/models/{model_name}_int8.engine \\")
    print(f"    --int8 \\")
    print(f"    --fp16 \\")
    print(f"    --workspace=4096 \\")
    print(f"    --verbose")
    print()
    print(f"# 3. 验证 TensorRT 引擎")
    print(f"trtexec --loadEngine=/home/gemini/models/{model_name}_int8.engine")
    print()
    print("✅ 预期结果:")
    print(f"  - 模型大小: ~{_estimate_int8_size(onnx_path):.1f} GB (INT8)")
    print(f"  - 推理延迟: ~20-40ms (NUC RTX 4060)")
    print(f"  - 显存占用: ~3-5 GB")
    print("=" * 60)


def _estimate_int8_size(onnx_path):
    """估算 INT8 模型大小（约为 ONNX 的 1/4）"""
    size_mb = os.path.getsize(onnx_path) / 1024**2
    return size_mb / 4 / 1024  # MB → GB, FP32→INT8=4x压缩


def main():
    parser = argparse.ArgumentParser(description="VLA 模型部署转换")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/smolvla/best.pt",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="smolvla",
        choices=["smolvla", "pi0"],
        help="模型类型",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="ONNX 输出路径（默认: <model>_deploy.onnx）",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="跳过 ONNX 验证",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.model}_deploy.onnx"

    print("=" * 60)
    print("Step 7 实验：VLA 模型部署转换")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    print(f"\n加载 checkpoint: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print(f"❌ 未找到 {args.ckpt}")
        print(f"  请先训练模型: uv run python codes/step5_smolvla/train.py")
        sys.exit(1)

    model = SmolVLA().to(device)
    state = torch.load(args.ckpt, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  模型: {args.model} ({n_params:.1f}M 参数)")

    # 导出 ONNX
    export_onnx(model, args.output, device=device)

    # 验证 ONNX
    if not args.skip_verify:
        verify_onnx(args.output)

    # TensorRT 转换指令
    print_tensorrt_instructions(args.output, args.model)

    print(f"\n⏭️  下一步: 复制 ONNX 到 Gemini NUC 并运行 trtexec")


if __name__ == "__main__":
    main()
