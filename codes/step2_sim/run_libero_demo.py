#!/usr/bin/env python3
"""Step 2 实验：LIBERO 仿真环境验证 + 随机动作 Demo

跑通这个脚本确认 MuJoCo + LIBERO 环境可用。
后续 Step 3 会把随机动作替换为真正的 OpenVLA 模型推理。

用法：
    uv run python codes/step2_sim/run_libero_demo.py
"""

import numpy as np
from libero.libero import benchmark


def main():
    print("=" * 50)
    print("Step 2 实验：LIBERO 仿真环境验证")
    print("=" * 50)

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

    # 4. 随机动作推理循环
    print("\n[3/4] 运行随机动作推理循环 (20 steps)...")
    max_steps = 20
    for step in range(max_steps):
        # 随机动作（Step 3 会替换为 OpenVLA 模型输出）
        action = np.random.uniform(-0.1, 0.1, 7)
        action[6] = 1.0  # gripper 保持张开

        obs, reward, done, info = env.step(action)

        if step % 5 == 0 or done:
            d = action
            print(
                f"  Step {step:2d} | "
                f"Δxyz=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) | "
                f"gripper={d[6]:+.3f} | "
                f"done={done}"
            )

        if done:
            print(f"  ✅ 任务意外完成于 step {step}")
            break

    # 5. 验证结果
    print("\n[4/4] 验证结果")
    print("  ✅ MuJoCo + LIBERO 环境正常工作")
    print("  ✅ 随机动作可以驱动仿真")
    print("  ⏭️  下一步: Step 3 — 用真正的 OpenVLA 模型替换随机动作")

    env.close()
    print("\n实验完成！")


if __name__ == "__main__":
    main()
