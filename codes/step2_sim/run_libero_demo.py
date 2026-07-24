#!/usr/bin/env python3
"""Step 2 实验：LIBERO 仿真 — 大幅定向动作（截图区分度明显）

问题：随机动作下机械臂在桌面同一位置抖动，top-down 相机看不出变化。
方案：用大幅定向动作让机械臂在桌面不同区域之间移动，
      4 个截图位置覆盖桌面左/中/右，配合夹爪开/关。

用法：
    cd /develop/vla-course/codes
    source .venv/bin/activate
    python step2_sim/run_libero_demo.py
"""

import os
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
TASK_BDDL_NAME = task.bddl_file
TASK_LANG = task.language
BDDL_DIR = os.path.join(get_libero_path("bddl_files"), "libero_spatial")
TASK_BDDL_PATH = os.path.join(BDDL_DIR, TASK_BDDL_NAME)


def _save_frame(obs, out_dir, step):
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
    print(f"  📸 Step {step:3d} -> {fname}")


def main():
    print("=" * 50)
    print("Step 2 实验：LIBERO 仿真 — 大幅定向动作 Demo")
    print("=" * 50)

    out_dir = os.path.join(os.path.dirname(__file__), "_screenshots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n任务: {TASK_BDDL_NAME.replace('.bddl', '')[:60]}")
    print(f"指令: {TASK_LANG}")

    print("\n创建仿真环境...")
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
    print(f"动作维度: {env.action_spec[0].shape[0]}")

    # ═══════════════════════════════════════════════════════
    # 帧 0：初始场景 — 机械臂默认姿态（桌面中央偏后上方）
    # ═══════════════════════════════════════════════════════
    _save_frame(obs, out_dir, 0)

    # ═══════════════════════════════════════════════════════
    # 帧 20：大幅右移 — 机械臂移到桌面右侧
    # ═══════════════════════════════════════════════════════
    action = np.zeros(8)
    action[1] = -0.25   # Y 大幅右移
    action[2] = -0.15   # Z 下降（接近桌面）
    action[-1] = 1.0
    for _ in range(20):
        obs, reward, done, info = env.step(action)
    _save_frame(obs, out_dir, 20)

    # ═══════════════════════════════════════════════════════
    # 帧 40：大幅左移 — 机械臂从右侧横跨到桌面左侧
    # ═══════════════════════════════════════════════════════
    action = np.zeros(8)
    action[1] = 0.30    # Y 大幅左移（跨越整个桌面宽度）
    action[3] = 0.20    # 手腕 roll 旋转
    action[-1] = 1.0
    for _ in range(20):
        obs, reward, done, info = env.step(action)
    _save_frame(obs, out_dir, 40)

    # ═══════════════════════════════════════════════════════
    # 帧 60：向下靠近物体 + 夹爪关闭（模拟抓取）
    # ═══════════════════════════════════════════════════════
    action = np.zeros(8)
    action[0] = 0.15    # X 前伸
    action[1] = -0.15   # Y 回移中间
    action[2] = -0.10   # Z 继续下降
    action[-1] = -1.0   # 夹爪关闭
    for _ in range(20):
        obs, reward, done, info = env.step(action)
    _save_frame(obs, out_dir, 60)

    # ═══════════════════════════════════════════════════════
    # 帧 80：上提取物 + 夹爪保持关闭（模拟"抓起并举起"）
    # ═══════════════════════════════════════════════════════
    action = np.zeros(8)
    action[2] = 0.20    # Z 大幅上提
    action[-1] = -1.0   # 夹爪保持关闭
    for _ in range(20):
        obs, reward, done, info = env.step(action)
    _save_frame(obs, out_dir, 80)

    print(f"\n截图已保存: {out_dir}/")
    print("✅ MuJoCo + LIBERO 环境正常工作")
    print("⏭️ 下一步: Step 3 — OpenVLA 推理替换定向动作")
    env.close()
    print("实验完成！")


if __name__ == "__main__":
    main()
