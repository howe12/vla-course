#!/usr/bin/env python3
"""Step 5 实验：SmolVLA 模型评估 — LIBERO 基准测试

加载训练好的 SmolVLA 模型，在 LIBERO-Spatial/Object/Goal 三套件上评测。

用法（L40 上）：
    uv run python codes/step5_smolvla/eval_libero.py --ckpt outputs/smolvla/best.pt
"""

import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

try:
    import torch
    from libero.libero import benchmark
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: uv pip install torch libero pillow")
    sys.exit(1)

# 从 train.py 导入模型定义
sys.path.insert(0, str(Path(__file__).parent))
from train import SmolVLA


def evaluate_libero(model, device, suite_names=None, max_steps=30):
    """在 LIBERO 基准上评估模型

    Args:
        model: SmolVLA 模型
        device: torch device
        suite_names: 要评估的套件列表
        max_steps: 每个任务的最大步数

    Returns:
        dict: {suite_name: success_rate}
    """
    if suite_names is None:
        suite_names = ["libero_spatial", "libero_object", "libero_goal"]

    benchmark_dict = benchmark.get_benchmark_dict()
    results = {}

    for suite_name in suite_names:
        if suite_name not in benchmark_dict:
            print(f"  ⚠️  套件 {suite_name} 不可用，跳过")
            continue

        task_suite = benchmark_dict[suite_name]()
        n_tasks = min(10, task_suite.n_tasks)
        successes = 0

        print(f"\n  {suite_name} ({n_tasks} tasks):")

        for task_id in range(n_tasks):
            task = task_suite.get_task(task_id)
            env_args = {
                "task": task,
                "bddl_file": task.bddl_file,
                "camera_heights": 224,
                "camera_widths": 224,
            }
            env = task_suite.get_env(task_id=task_id, env_args=env_args)

            obs = env.reset()
            done = False

            for step in range(max_steps):
                # 图像预处理
                image = Image.fromarray(obs).convert("RGB")
                image_tensor = (
                    torch.from_numpy(
                        np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                # 模型推理
                with torch.no_grad():
                    action = model(image_tensor)  # (1, 7)

                obs, reward, done, info = env.step(action[0].cpu().numpy())

                if done:
                    break

            successes += int(done)
            status = "✅" if done else "❌"
            print(f"    Task {task_id:2d}: {status}  {task.name[:50]}")

            env.close()

        rate = successes / n_tasks
        results[suite_name] = rate
        print(f"    → 成功率: {successes}/{n_tasks} = {rate:.0%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SmolVLA LIBERO 评估")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/smolvla/best.pt",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--suites",
        type=str,
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal"],
        help="要评估的 LIBERO 套件",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 5 实验：SmolVLA 模型评估")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"\n❌ 未找到模型: {ckpt_path}")
        print(f"  请先运行训练: uv run python codes/step5_smolvla/train.py")
        print(f"  或指定其他路径: --ckpt <path>")
        sys.exit(1)

    print(f"\n加载模型: {ckpt_path}")
    model = SmolVLA().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)

    # 兼容不同保存格式
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  参数: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 评估
    print(f"\n开始评估: {args.suites}")
    results = evaluate_libero(model, device, args.suites)

    # 汇总
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    for suite, rate in results.items():
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {suite:25s}  {bar}  {rate:.0%}")

    avg = sum(results.values()) / len(results) if results else 0
    print(f"  {'平均':25s}  {'█' * int(avg * 20) + '░' * (20 - int(avg * 20))}  {avg:.0%}")

    # 对比参考值
    print("\n参考对比:")
    print(f"  OpenVLA zero-shot:  ~55%")
    print(f"  SmolVLA (本文):     {avg:.0%}")
    print(f"  RT-2 (55B):         ~70% (参考)")

    print(f"\n⏭️  下一步: Step 6 — Pi0 扩散 VLA 训练")


if __name__ == "__main__":
    main()
