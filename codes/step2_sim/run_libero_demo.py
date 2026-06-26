#!/usr/bin/env python3
"""Step 2 实验：LIBERO 仿真环境验证 + 动画截图

跑通这个脚本确认 MuJoCo + LIBERO 环境可用，并自动保存仿真截图。
后续 Step 3 会把随机动作替换为真正的 OpenVLA 模型推理。

用法：
    cd /develop/vla-course/codes
    source .venv/bin/activate
    python step2_sim/run_libero_demo.py

输出：
    step2_sim/_screenshots/libero_step_*.png

注意：首次运行需下载 LIBERO 资产（~586文件），请确保代理可用。
"""

import os
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING

# ── 从 benchmark 获取任务信息 ──
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
TASK_BDDL_NAME = task.bddl_file  # 例如 "pick_up_xxx.bddl"
TASK_LANG = task.language

# ── 拼接完整 BDDL 路径 ──
BDDL_DIR = os.path.join(get_libero_path("bddl_files"), "libero_spatial")
TASK_BDDL_PATH = os.path.join(BDDL_DIR, TASK_BDDL_NAME)


def main():
    print("=" * 50)
    print("Step 2 实验：LIBERO 仿真环境验证")
    print("=" * 50)

    out_dir = os.path.join(os.path.dirname(__file__), "_screenshots")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 任务信息
    task_short = TASK_BDDL_NAME.replace(".bddl", "")[:60]
    print(f"\n[1/4] 任务: {task_short}...")
    print(f"  语言指令: {TASK_LANG}")

    # 2. 创建仿真环境
    print("\n[2/4] 创建仿真环境...")
    env = TASK_MAPPING["libero_tabletop_manipulation"](
        bddl_file_name=TASK_BDDL_PATH,
        robots=["Panda"],
        has_offscreen_renderer=True,
        has_renderer=False,
        camera_heights=224,
        camera_widths=224,
        render_gpu_device_id=-1,
    )

    obs = env.reset()
    print(f"  观测 keys: {list(obs.keys())}")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"    {k}: shape={v.shape}")
    print(f"  动作维度: {env.action_spec[0].shape[0]}")

    _save_frame(obs, out_dir, 0)

    # 3. 随机动作推理循环
    print("\n[3/4] 运行随机动作 (80 steps)...")
    capture_steps = [20, 40, 60, 80]

    for step in range(1, 81):
        action = np.random.uniform(-0.1, 0.1, 8)
        action[-1] = 1.0
        obs, reward, done, info = env.step(action)

        if step in capture_steps:
            _save_frame(obs, out_dir, step)

        if done:
            print(f"  ✅ 任务意外完成于 step {step}")
            break

    # 4. 结果
    print(f"\n[4/4] 截图已保存: {out_dir}/")
    print("  ✅ MuJoCo + LIBERO 环境正常工作")
    print("  ⏭️  下一步: Step 3 — OpenVLA 推理替换随机动作")
    env.close()
    print("\n实验完成！")


def _save_frame(obs, out_dir, step):
    """保存 agentview 相机画面"""
    key = "agentview_image"
    if key not in obs:
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3:
                key = k
                break
    img = obs[key]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    fname = f"libero_step_{step:03d}.png"
    Image.fromarray(img).save(os.path.join(out_dir, fname))
    print(f"  📸 Step {step:3d} — {fname}")


if __name__ == "__main__":
    main()
