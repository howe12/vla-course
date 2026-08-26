#!/usr/bin/env python3
"""Sagittarius SGR532 轨迹关节角曲线 (03-1 数据采集)
把采集的 CSV 轨迹画成 关节角-时间 曲线，一眼看清 8 个关节随时间的走动，
是采集后检查数据有没有"动作语义"的直观手段。
用法: python sgr532_plot.py [traj.csv]   -> 输出 sgr532_traj_plot.png
"""
import os, sys, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    cand = sorted(glob.glob(os.path.join(HERE, "sgr532_traj_*.csv")))
    if not cand:
        sys.exit("没找到轨迹 CSV")
    csv_path = cand[-1]

cells = {}
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        for k, v in row.items():
            cells.setdefault(k, []).append(float(v))
t = np.array(cells.pop("t"))
names = ["J1","J2","J3","J4","J5","J6","GL","GR"]

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = plt.cm.tab10(np.linspace(0, 0.65, len(names)))
for name, col, c in zip(names, [np.array(cells[k]) for k in names], colors):
    ax.plot(t, np.degrees(col), label=name, color=c, lw=1.4)
ax.set_xlabel("时间 (s)"); ax.set_ylabel("关节角 (度)")
ax.set_title("SGR532 采集轨迹 8 关节角-时间曲线")
ax.grid(alpha=0.3, ls=":")
ax.legend(ncol=4, fontsize=8, loc="upper right")
fig.tight_layout()
out = os.path.join(HERE, "sgr532_traj_plot.png")
fig.savefig(out, dpi=130)
print(f"[图] {out}")
print(f"[统计] {len(t)} 步, 时长 {t[-1]-t[0]:.2f} s")
print("=== SGR532 轨迹曲线完成 ===")