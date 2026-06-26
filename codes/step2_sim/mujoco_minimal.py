#!/usr/bin/env python3
"""MuJoCo 入门 1/3：最简物理仿真 — 小球自由落体（+ 渲染截图）

用 30 行代码理解 MuJoCo 的核心循环：
  XML → mjModel（"图纸"）→ mjData（"快照"）→ mj_step（推进）

同时保存 4 帧渲染截图，展示小球从高处落下的过程。

运行：
    cd /develop/vla-course/codes && source .venv/bin/activate
    python step2_sim/mujoco_minimal.py
"""

import os, mujoco, numpy as np
from PIL import Image

xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="ball" pos="0 0 1.5">
      <joint type="free"/>
      <geom type="sphere" size="0.1" rgba="1 0.3 0.3 1" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

out_dir = os.path.join(os.path.dirname(__file__), "_screenshots")
os.makedirs(out_dir, exist_ok=True)

print(f"模型: {model.nq} 自由度, dt={model.opt.timestep}s, g=({model.opt.gravity[2]:.2f})")
print(f"小球初始高度: {data.qpos[2]:.2f}m\n")

capture_steps = [0, 60, 120, 180]  # 捕获 4 个时刻

for step in range(200):
    mujoco.mj_step(model, data)

    if step in capture_steps:
        renderer.update_scene(data)
        frame = renderer.render()
        fname = f"mujoco_ball_{step:03d}.png"
        fpath = os.path.join(out_dir, fname)
        Image.fromarray(frame).save(fpath)
        ball_z = data.qpos[2]
        print(f"  📸 step {step:3d} → {fname}  (球高={ball_z:.3f}m)")

    if data.qpos[2] < 0.15:
        last_step = step
        break
else:
    last_step = 199

# 最后一张：球落地后
renderer.update_scene(data)
frame = renderer.render()
fname = "mujoco_ball_landed.png"
fpath = os.path.join(out_dir, fname)
Image.fromarray(frame).save(fpath)
print(f"  📸 step {last_step:3d} → {fname}  (球高={data.qpos[2]:.3f}m) ✅ 落地")

print(f"\n截图保存在: {out_dir}/")
print(f"🔑 200 步 × 0.002s = 0.4s 仿真时间")
