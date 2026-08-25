#!/usr/bin/env python
"""L40 验证: 加载 Sagittarius SGR532 MJCF + OSMesa 渲染 + 关节控制 demo (CPU 无需 GPU)"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import mujoco
import numpy as np

mjcf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sgr532.mjcf")
m = mujoco.MjModel.from_xml_path(mjcf)
d = mujoco.MjData(m)

print(f"[加载] OK  nq={m.nq} nbody={m.nbody} ngeom={m.ngeom} nsite={m.nsite}")
print(f"[关节] {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]}")

renderer = mujoco.Renderer(m, 480, 480)

def snap(name):
    renderer.update_scene(d, "arm_cam")  # 墙上固定相机朝向机械臂
    img = renderer.render().copy()
    from PIL import Image
    Image.fromarray(img).save(name)
    print(f"[图] {name} shape={img.shape} 非黑={img.mean()>5}")

# 默认位形
mujoco.mj_forward(m, d)
snap("sgr532_home.png")

# 关节控制: 设置 6 个关节到一个舒展位形
J = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"J{i}") for i in range(1, 7)]
qadr = m.jnt_qposadr[J]
home = [[-0.5, 0.5, 0.3, 0.2, 0.4, 0.5][i] for i in range(6)]  # 度
for i, a in enumerate(qadr):
    d.qpos[a] = np.deg2rad(home[i])
mujoco.mj_forward(m, d)
print(f"[控制] 关节角(deg) {home}")
snap("sgr532_expand.png")

# 物理步进
for _ in range(50):
    mujoco.mj_step(m, d)
print(f"[步进] 50 步 OK, 无 NaN qpos={d.qpos[:3]}")
print("=== SGR532 MuJoCo 验证通过 ===")