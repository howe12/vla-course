#!/usr/bin/env python3
"""Step 2 实验：LIBERO 仿真环境验证 + 动画截图

跑通这个脚本确认 MuJoCo + LIBERO 环境可用，并自动保存仿真截图。
后续 Step 3 会把随机动作替换为真正的 OpenVLA 模型推理。

用法：
    uv run python codes/step2_sim/run_libero_demo.py

输出：
    codes/step2_sim/_screenshots/libero_step_*.png  (4 帧观测截图)
"""

import os
import numpy as np
from PIL import Image
from libero.libero import benchmark


def main():
    print("=" * 50)
    print("Step 2 实验：LIBERO 仿真环境验证")
    print("=" * 50)

    # ---- 截图输出目录 ----
    out_dir = os.path.join(os.path.dirname(__file__), "_screenshots")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 加载 LIBERO-Spatial 基准
    print("\n[1/4] 加载 LIBERO-Spatial...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(0)

    print(f"  任务名称: {task.name}")
    print(f"  语言指令: {task.language_instruction}")

    # 2. 创建 gym 环境
    print("\n[2/4] 创建仿真环境...")
    env_args = {
        "task": task,
        "bddl_file": task.bddl_file,
        "camera_heights": 224,
        "camera_widths": 224,
    }
    env = task_suite.get_env(task_id=0, env_args=env_args)

    # 3. 查看观测和动作空间
    obs = env.reset()
    print(f"  观测 shape: {obs.shape}  (应为 224, 224, 3)")
    print(f"  动作维度:   {env.action_space.shape[0]}  (应为 7)")

    # ---- 保存初始帧 ----
    Image.fromarray(obs).save(os.path.join(out_dir, "libero_step_000.png"))
    print(f"  📸 保存初始帧: step_000 (RGB {obs.shape})")

    # 4. 随机动作推理循环（每 20 步抓一帧，共 4 帧）
    print("\n[3/4] 运行随机动作推理循环 (80 steps，每 20 步截图)...")
    max_steps = 80
    capture_steps = [0, 20, 40, 60]  # 含初始帧共 5 帧

    for step in range(1, max_steps + 1):
        # 随机动作（Step 3 会替换为 OpenVLA 模型输出）
        action = np.random.uniform(-0.1, 0.1, 7)
        action[6] = 1.0  # gripper 保持张开

        obs, reward, done, info = env.step(action)

        if step in capture_steps:
            fname = f"libero_step_{step:03d}.png"
            Image.fromarray(obs).save(os.path.join(out_dir, fname))
            print(f"  📸 Step {step:3d} — 保存截图: {fname}")

        if done:
            print(f"  ✅ 任务意外完成于 step {step}")
            break

    # 5. 验证结果
    print("\n[4/4] 验证结果")
    print("  ✅ MuJoCo + LIBERO 环境正常工作")
    print("  ✅ 随机动作可以驱动仿真")
    print(f"  📸 截图已保存到: {out_dir}/")
    print("  ⏭️  下一步: Step 3 — 用真正的 OpenVLA 模型替换随机动作")

    env.close()
    print("\n实验完成！")


if __name__ == "__main__":
    main()
