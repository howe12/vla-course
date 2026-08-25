#!/usr/bin/env python
"""L40 VLA 仿真课程 - 环境自检 (CPU 部分, 无需 GPU)
验证: MuJoCo 编译 + 物理步进 + OSMesa 离屏渲染 + lerobot 可导入
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. MuJoCo 编译
import mujoco
xml = """<mujoco model="envcheck"><option timestep="0.02"/>
<worldbody>
  <geom type="plane" size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
  <body name="ball" pos="0 0 1"><freejoint/><geom type="sphere" size="0.08" rgba="1 0.2 0.2 1"/></body>
</worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
print(f"[1] MuJoCo 编译 OK  nq={m.nq} njnt={m.njnt} nbody={m.nbody}")

# 2. 物理步进
for _ in range(10):
    mujoco.mj_step(m, d)
print(f"[2] 物理步进 OK  ball qpos[2]={d.qpos[2]:.3f}")

# 3. OSMesa 离屏渲染 (无需 GPU 设备)
renderer = mujoco.Renderer(m, 320, 240)
renderer.update_scene(d)
img = renderer.render().copy()
print(f"[3] OSMesa 渲染 OK  shape={img.shape} 非黑={img.mean()>5}")

# 4. torch (cuda 可供后续参考)
import torch
print(f"[4] torch {torch.__version__}  cuda_available={torch.cuda.is_available()} (推理/训练需 GPU 开启)")

# 5. lerobot
import lerobot
print(f"[5] lerobot 可导入")

print("\n=== CPU 自检通过, L40 环境就绪 ===")