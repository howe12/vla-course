#!/usr/bin/env python3
"""LIBERO 入门 2/2：动作空间实验 — 理解每维动作的含义

用一个任务反复测试不同的动作值，观察：
  1. Δx / Δy / Δz 分别让末端往哪个方向移动
  2. gripper 的 -1(关) / +1(开) 如何影响夹爪
  3. 动作过大会发生什么（限位？穿模？）
  4. 一张截图 vs 一段轨迹的区别

运行：
    cd /develop/vla-course/codes
    source .venv/bin/activate
    python step2_sim/libero_action_test.py
"""

import os
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING

out_dir = os.path.join(os.path.dirname(__file__), "_screenshots")
os.makedirs(out_dir, exist_ok=True)

benchmark_dict = benchmark.get_benchmark_dict()
suite = benchmark_dict["libero_spatial"]()
task = suite.get_task(0)

bddl_dir = os.path.join(get_libero_path("bddl_files"), "libero_spatial")
bddl_path = os.path.join(bddl_dir, task.bddl_file)

env = TASK_MAPPING["libero_tabletop_manipulation"](
    bddl_file_name=bddl_path,
    robots=["Panda"],
    has_offscreen_renderer=True,
    has_renderer=False,
    camera_heights=224,
    camera_widths=224,
    render_gpu_device_id=-1,
)

print("=" * 55)
print("LIBERO 入门 2/2：动作空间实验")
print("=" * 55)
print(f"\n任务: {task.language}")
print(f"动作维度: {env.action_spec[0].shape[0]}")
print(f"""
动作空间语义（8 维）:
  [0] Δx    — 末端 X 方向增量（前后）
  [1] Δy    — 末端 Y 方向增量（左右）
  [2] Δz    — 末端 Z 方向增量（上下）
  [3] Δroll — 绕 X 轴旋转
  [4] Δpitch— 绕 Y 轴旋转  
  [5] Δyaw  — 绕 Z 轴旋转
  [6] gripper_ctrl — 夹爪控制（正=开，负=关）
  [7] (保留，通常设 0)
""")

def save_frame(obs, label):
    img = obs["agentview_image"]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    fname = f"action_{label}.png"
    Image.fromarray(img).save(os.path.join(out_dir, fname))
    print(f"  📸 保存 {fname}")

# ═══════════════════════════════════════════
# 实验 1：向前移动（Δx = +0.3）
# ═══════════════════════════════════════════
print(f"\n{'='*55}")
print("实验 1：向前移动（ΔX = +0.3，重复 20 步）")
print(f"{'='*55}")

env.reset()
action = np.zeros(8)
action[0] = 0.3   # X 前移
action[-1] = 1.0  # gripper 开
for _ in range(20):
    obs, reward, done, info = env.step(action)
save_frame(obs, "forward")
ee = obs["robot0_eef_pos"]
print(f"  末端位置: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")
print(f"  💡 Δx=+0.3 × 20步 = 末端向前移动了 ~{0.3*20*0.01:.2f}m")

# ═══════════════════════════════════════════
# 实验 2：向右移动（Δy = -0.3）
# ═══════════════════════════════════════════
print(f"\n{'='*55}")
print("实验 2：向右移动（ΔY = -0.3，重复 20 步）")
print(f"{'='*55}")

env.reset()
action = np.zeros(8)
action[1] = -0.3  # Y 右移
action[-1] = 1.0
for _ in range(20):
    obs, reward, done, info = env.step(action)
save_frame(obs, "right")
ee = obs["robot0_eef_pos"]
print(f"  末端位置: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")
print(f"  💡 Δy=-0.3 → 末端在 Y 轴负方向（从顶视图看是向右）")

# ═══════════════════════════════════════════
# 实验 3：向下移动（Δz = -0.3）
# ═══════════════════════════════════════════
print(f"\n{'='*55}")
print("实验 3：向下移动（ΔZ = -0.3，重复 20 步）")
print(f"{'='*55}")

env.reset()
action = np.zeros(8)
action[2] = -0.3  # Z 下降
action[-1] = 1.0
for _ in range(20):
    obs, reward, done, info = env.step(action)
save_frame(obs, "down")
ee = obs["robot0_eef_pos"]
print(f"  末端位置: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")
print(f"  💡 Δz=-0.3 → 末端下降（Z 轴向上为正）")

# ═══════════════════════════════════════════
# 实验 4：夹爪开 vs 关
# ═══════════════════════════════════════════
print(f"\n{'='*55}")
print("实验 4：夹爪控制（gripper 开 vs 关）")
print(f"{'='*55}")

env.reset()
action = np.zeros(8)
action[-1] = -1.0  # gripper 关
for _ in range(10):
    obs, reward, done, info = env.step(action)
save_frame(obs, "gripper_close")
gripper = obs["robot0_gripper_qpos"]
print(f"  夹爪位置: {gripper}  (关闭状态, gripper_ctrl=-1)")

env.reset()
action = np.zeros(8)
action[-1] = 1.0   # gripper 开
for _ in range(10):
    obs, reward, done, info = env.step(action)
save_frame(obs, "gripper_open")
gripper = obs["robot0_gripper_qpos"]
print(f"  夹爪位置: {gripper}  (打开状态, gripper_ctrl=+1)")
print(f"  💡 gripper_ctrl: 正→开, 负→关，中间值对应部分开合")

# ═══════════════════════════════════════════
# 实验 5：动作过大 → 限位
# ═══════════════════════════════════════════
print(f"\n{'='*55}")
print("实验 5：动作过大 → 观察限位行为")
print(f"{'='*55}")

env.reset()
action = np.zeros(8)
action[0] = 2.0    # ΔX 大幅超出范围
action[-1] = 1.0
ee_before = obs["robot0_eef_pos"].copy()
for _ in range(20):
    obs, reward, done, info = env.step(action)
ee_after = obs["robot0_eef_pos"]
print(f"  末端移动量: Δ=({ee_after[0]-ee_before[0]:.3f}, "
      f"{ee_after[1]-ee_before[1]:.3f}, {ee_after[2]-ee_before[2]:.3f})")
print(f"  💡 即使 action[0]=2.0（超大），实际位移也只有 ~0.06m")
print(f"     因为环境内部有动作限幅（clipping），不会让机械臂飞出去")

env.close()

print(f"\n{'='*55}")
print("✅ 动作空间实验完成！")
print(f"   截图保存在: {out_dir}/")
print(f"\n🔑 带走:")
print(f"   动作 = 末端增量（不是绝对位置！）")
print(f"   Δx/Δy/Δz 控制空间移动，Δroll/Δpitch/Δyaw 控制旋转")
print(f"   gripper_ctrl: +1=开, -1=关")
print(f"   环境内部对过大动作有限幅保护")
print(f"   ⏭️  第 3 章 VLA 模型输出的就是这 7 个增量值")
