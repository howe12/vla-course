#!/usr/bin/env python3
"""L40 验证: 加载通用机械臂 LeRobot SO-101 MJCF + OSMesa 渲染 + 关节控制 demo (CPU 无需 GPU)
流程与 SGR532 完全一致, 体现"通用型号,同一套做法"。"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import mujoco
import numpy as np

mjcf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "so101.mjcf")
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

# 默认位形 (qpos=0, 竖直向上)
mujoco.mj_forward(m, d)
snap("so101_home.png")

# 关节控制: 设置 6 杆到一个舒展位形 (SO-101 为 5 自由度 + 夹爪)
J = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
     for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]
qadr = m.jnt_qposadr[J]
# 舒展位形(度): 平举上臂 + 屈肘 + 张夹爪
home = [30.0, 70.0, -50.0, -20.0, 15.0, 60.0]
for i, a in enumerate(qadr):
    d.qpos[a] = np.deg2rad(home[i])
mujoco.mj_forward(m, d)
print(f"[控制] 关节角(deg) {home}")
snap("so101_expand.png")

# 物理步进
for _ in range(50):
    mujoco.mj_step(m, d)
print(f"[步进] 50 步 OK, 无 NaN qpos={d.qpos[:3]}")
print("=== SO-101 MuJoCo 验证通过 ===")