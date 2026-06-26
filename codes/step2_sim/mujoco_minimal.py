#!/usr/bin/env python3
"""MuJoCo 入门 1/3：最简物理仿真 — 小球自由落体

用 30 行代码理解 MuJoCo 的核心循环：
  XML 定义世界 → mjModel（"图纸"）→ mjData（"快照"）→ mj_step（推进）

运行：
    cd /develop/vla-course/codes
    source .venv/bin/activate
    python step2_sim/mujoco_minimal.py
"""

import mujoco
import numpy as np

# ═══ 1. 定义世界：用 XML 字符串描述场景 ═══
xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="ball" pos="0 0 1.5">
      <joint type="free"/>                          <!-- 自由体（6 DOF） -->
      <geom type="sphere" size="0.1" rgba="1 0.3 0.3 1" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

# ═══ 2. 加载模型：XML → mjModel（编译为物理"图纸"） ═══
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)  # mjData：运行时的"状态快照"

print(f"模型有 {model.nq} 个位置自由度, {model.nv} 个速度自由度")
print(f"物理步长 dt={model.opt.timestep:.4f}s  重力 g=({model.opt.gravity[0]},{model.opt.gravity[1]},{model.opt.gravity[2]})")
print()

# ═══ 3. 仿真循环：mj_step 推进物理世界 ═══
print(f"{'Step':>5}  {'time':>8}  {'ball_z':>8}  {'ball_vz':>10}")
print("-" * 45)

for step in range(200):
    # ── 核心：万物皆在 mj_step ──
    mujoco.mj_step(model, data)

    # ── 读取状态：mjData 是活的状态快照 ──
    ball_z = data.qpos[2]     # 球的 Z 坐标（第 3 个位置分量）
    ball_vz = data.qvel[2]    # 球的 Z 速度

    if step % 20 == 0:
        print(f"{step:5d}  {data.time:8.4f}  {ball_z:8.4f}  {ball_vz:10.4f}")

    # 球落地后停止
    if ball_z < 0.15 and ball_vz > -0.01:
        print(f"\n✅ 球在第 {step} 步落地 (t={data.time:.3f}s)")
        print(f"   理论落地时间: sqrt(2*1.5/9.81) ≈ {np.sqrt(2*1.5/9.81):.3f}s ✓")
        break

print(f"\n🔑 核心公式: 1 次 mj_step = 物理时间推进 {model.opt.timestep}s")
