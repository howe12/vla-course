#!/usr/bin/env python3
"""Sagittarius SGR532 轨迹回放验证 (03-1 数据采集)
读键盘遥操作/采集存的 CSV 轨迹，在 MuJoCo 里逐帧重放并渲染成 GIF，
用来检查"采到的那条轨迹到底像不像样"，是数据采集后必做的第一步核验。
用法:
  python sgr532_replay.py                 # 回放默认最新轨迹 sgr532_traj_*.csv
  python sgr532_replay.py sgr532_traj_x.csv
"""
import os, sys, glob, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["MUJOCO_GL"] = "osmesa"          # 无显示器云服务器离屏渲染
import mujoco
from PIL import Image

# ---- 选轨迹文件 ----
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    cand = sorted(glob.glob(os.path.join(HERE, "sgr532_traj_*.csv")))
    if not cand:
        sys.exit("没找到轨迹 CSV，先用 sgr532_teleop.py 采一条")
    csv_path = cand[-1]
print(f"[轨迹] {csv_path}")

# ---- 读 CSV ----
rows = []
with open(csv_path, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append((float(row["t"]), [float(row[k]) for k in
                     ["J1","J2","J3","J4","J5","J6","GL","GR"]]))
t, traj = zip(*rows)
traj = np.array(traj)
print(f"[读取] {len(traj)} 步, 时长 {(t[-1]-t[0]):.2f} s, 8 关节(6摆位+2夹爪)")

# ---- 加载模型重放 ----
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "sgr532.mjcf"))
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, 480, 480)
frames = []
STRIDE = max(1, len(traj) // 200)          # 最多渲染 ~200 帧，控制 GIF 体积
for i in range(0, len(traj), STRIDE):
    d.qpos[:8] = traj[i]
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, "arm_cam")
    frames.append(Image.fromarray(renderer.render()))

out = os.path.join(HERE, "sgr532_replay.gif")
try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = Image.ANTIALIAS
small = [f.resize((320, 240), resample) for f in frames]
small[0].save(out, save_all=True, append_images=small[1:], duration=30, loop=0)
print(f"[回放] {len(frames)} 帧 -> {out}  {os.path.getsize(out)//1024}KB")

# ---- 采集质量摘要 ----
print("[质量] 各关节转角范围(度):")
for k, col in zip(["J1","J2","J3","J4","J5","J6","GL","GR"], traj.T):
    print(f"  {k:3s} min={np.degrees(col.min()):7.2f} max={np.degrees(col.max()):7.2f} "
          f"span={np.degrees(col.max()-col.min()):6.2f}")
print("=== SGR532 轨迹回放验证完成 ===")