#!/usr/bin/env python3
"""Sagittarius SGR532 运动控制 demo (02-4)
在 MuJoCo 里用 position 伺服让机械臂"流着走"——从 home 到探出、低下、抬起、再收回，
每个关节的目标角用 smoothstep 插值，位置伺服平滑跟到位，并渲染成 GIF 演示臂连续动起来。
CPU 即可运行，无需 GPU (OSMesa 离屏渲染)。
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
import mujoco, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 加载(带 actuator 的控制版模型) ----
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "sgr532_motion.mjcf"))
d = mujoco.MjData(m)
assert m.nu == 8, f"期望 8 个 actuator, 实际 {m.nu}"

J = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"J{i}") for i in range(1, 7)]
GL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "GL")
GR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "GR")
qadr = m.jnt_qposadr[[*J, GL, GR]]

# ---- 姿态序列 (rad) [J1..J6, GL, GR], 夹爪 GL/GR 同号张开、异号闭合 ----
home  = np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.000,  0.000])
reach = np.array([ 0.5, -0.9,  0.8,  0.4, -0.5,  0.6,  0.004,  0.004])  # 探出, 夹爪张开
grasp = np.array([ 0.3,  0.5, -0.6, -0.4,  0.5,  0.3, -0.004, -0.004])  # 低下, 夹爪闭合
lift  = np.array([ 0.4, -1.0,  1.2,  0.7, -0.7,  0.9, -0.004, -0.004])  # 抬起, 夹爪保持闭合
motion = [home, reach, grasp, lift, home]   # home -> reach -> grasp -> lift -> home

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)

# ---- 初始位置 ----
mjcf = np.zeros(m.nq)
mjcf[list(qadr)] = home
d.qpos[:] = mjcf
mujoco.mj_forward(m, d)
for i, a in enumerate(qadr):
    d.ctrl[i] = home[i]                      # ctrl = 目标关节角(rad)

renderer = mujoco.Renderer(m, 480, 480)
frames = []
SEG_STEPS = 300   # 每段仿真步数
SAMPLE = 12       # 每采样步渲染一帧

def snap():
    renderer.update_scene(d, "arm_cam")
    frames.append(Image.fromarray(renderer.render()))

print("[PID] position 伺服 kp=100~120, kv=15~18, 关节注入 ctrl 目标角")
for seg, (cur, nxt) in enumerate(zip(motion[:-1], motion[1:])):
    for s in range(SEG_STEPS):
        u = smoothstep(0, SEG_STEPS - 1, s)      # 0 -> 1
        d.ctrl[:] = cur + (nxt - cur) * u        # 目标沿 smoothstep 向下一姿态插值
        mujoco.mj_step(m, d)
        if s % SAMPLE == 0:
            snap()
    # 该段结束: 各关节实际角应接近目标
    e = abs(d.qpos[list(qadr)] - nxt).max()
    print(f"[段{seg+1}] {chr(65+seg)}({'->'.join(str(x) for x in nxt.round(2))}) 伺服误差 {e:.4f} rad")

# ---- 合并 GIF ----
gif = os.path.join(HERE, "sgr532_motion.gif")
try:
    resample = Image.Resampling.LANCZOS     # Pillow >= 9.1
except AttributeError:
    resample = Image.ANTIALIAS              # 老版本
small = [f.resize((320, 240), resample) for f in frames]
small[0].save(gif, save_all=True, append_images=small[1:], duration=70, loop=0)
print(f"[GIF] {len(frames)} 帧 -> {gif}  {os.path.getsize(gif)//1024}KB")
print("=== SGR532 运动控制 demo 完成 ===")