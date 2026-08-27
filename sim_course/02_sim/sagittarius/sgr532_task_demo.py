#!/usr/bin/env python3
# 桌面双杯抓取-放置 任务场景演示: 随机初始布局 + 物理落定 + 双相机出图
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

XML = HERE / "sgr532_task.mjcf"
m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)

rng = np.random.default_rng(0)

def place_cup(name, xr, yr):
    ji = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    qr = m.jnt_qposadr[ji]
    x = rng.uniform(*xr); y = rng.uniform(*yr)
    d.qpos[qr:qr+3] = [x, y, 0.20]
    ang = rng.uniform(0, 6.28)
    d.qpos[qr+3:qr+7] = [0, 0, np.sin(ang/2), np.cos(ang/2)]

# 红杯蓝杯分区摆放, 保证互不重叠、都在桌面(x∈[0.14,0.50], y∈[-0.16,0.16])
place_cup('cup_red',  xr=(0.17, 0.24), yr=(-0.09, -0.02))
place_cup('cup_blue', xr=(0.25, 0.30), yr=(0.02, 0.09))

# 让 freejoint 物体在重力下落到桌面
for _ in range(500):
    mujoco.mj_step(m, d)
assert d.qacc[0] == d.qacc[0], "NaN after settle"

print("=== 落定后物体位置(桌面 z=0.14, 杯心 z≈0.18) ===")
for name in ['cup_red', 'cup_blue']:
    ji = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    qr = m.jnt_qposadr[ji]
    print(f"  {name:9s}{np.round(d.qpos[qr:qr+3],3)}  z={d.qpos[qr+2]:.3f}")
print("接触对数:", d.ncon)
print("任务描述: 把红色杯子放到绿色托盘  (task id 对应语言在数据 meta/tasks.jsonl)")

# 双相机渲染
ren = mujoco.Renderer(m, 480, 480)
out = {}
for cam in ["cam_top", "arm_cam"]:
    ren.update_scene(d, cam)
    out[cam] = ren.render().copy()
ren.close()

from PIL import Image
Image.fromarray(out["cam_top"]).save("sgr532_task_top.png")
Image.fromarray(out["arm_cam"]).save("sgr532_task_arm.png")
print("已输出 sgr532_task_top.png (顶视) / sgr532_task_arm.png (臂随)")