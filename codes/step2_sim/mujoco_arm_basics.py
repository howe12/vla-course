#!/usr/bin/env python3
"""MuJoCo 入门 2/3：关节控制 — 渲染机械臂姿态对比

在"收拢姿态"和"展开姿态"两个时刻各渲染一帧，
直观看到机械臂从紧凑 → 前伸的形态变化。

运行：
    cd /develop/vla-course/codes && source .venv/bin/activate
    python step2_sim/mujoco_arm_basics.py
"""

import os, mujoco, numpy as np
from PIL import Image

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models", "widowx_arm.xml",
)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

out_dir = os.path.join(os.path.dirname(__file__), "_screenshots")
os.makedirs(out_dir, exist_ok=True)

pose_tucked = np.array([0.0, -0.3, -0.5, 0.0, 0.3, 0.0])
pose_reach  = np.array([0.2, -1.2, -1.5, 0.0, -0.5, 0.0])

print("=" * 50)
print("MuJoCo 入门 2/3：关节控制（渲染版）")
print("=" * 50)

# ── 姿态 A：收拢 ──
mujoco.mj_resetData(model, data)
data.ctrl[:] = pose_tucked
print(f"\n收拢姿态 ctrl: {pose_tucked}")
for _ in range(200):
    mujoco.mj_step(model, data)

renderer.update_scene(data, camera="wrist_cam")
frame = renderer.render()
fpath = os.path.join(out_dir, "mujoco_arm_tucked.png")
Image.fromarray(frame).save(fpath)
ee = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")]
print(f"  📸 {fpath}")
print(f"  末端位置: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")

# ── 姿态 B：展开前伸 ──
mujoco.mj_resetData(model, data)
data.ctrl[:] = pose_reach
print(f"\n展开姿态 ctrl: {pose_reach}")
for _ in range(200):
    mujoco.mj_step(model, data)

renderer.update_scene(data, camera="wrist_cam")
frame = renderer.render()
fpath = os.path.join(out_dir, "mujoco_arm_reach.png")
Image.fromarray(frame).save(fpath)
ee = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")]
print(f"  📸 {fpath}")
print(f"  末端位置: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")
print(f"  末端位移: {np.linalg.norm(ee - np.array([0,0,0.61])):.2f}m")

print(f"\n✅ 截图保存在: {out_dir}/")
print(f"🔑 ctrl 变化 → qpos 跟踪 → 末端位移 → VLA 控制闭环")
