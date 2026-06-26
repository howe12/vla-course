#!/usr/bin/env python3
"""MuJoCo 入门 2/3：关节控制 — 理解 ctrl → qpos 跟踪

核心概念：
  data.ctrl[i]  = 你设置的目标位置（类似 VLA 输出的动作）
  data.qpos[idx] = 物理仿真后的实际关节角（PD 跟踪到位）
  mj_step 推进 dt=0.002s，反复调用 → 物理世界演化

VLA 关系：
  VLA 每 ~40ms 输出 6 个目标关节角 → data.ctrl → 20 次 mj_step → 到位

运行：
    cd /develop/vla-course/codes && source .venv/bin/activate
    python step2_sim/mujoco_arm_basics.py
"""

import os, mujoco, numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models", "widowx_arm.xml",
)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("=" * 55)
print("MuJoCo 入门 2/3：关节控制")
print("=" * 55)

# ═══ 1. 模型结构 ═══
print(f"\n📐 模型结构:")
print(f"   总 qpos = {model.nq}  (7 自由体 + 6 关节)")
print(f"   ctrl   = {model.nu}  (6 个位置执行器)")
print(f"   物理步长 = {model.opt.timestep}s")
print(f"   控制频率 = {1/(model.opt.timestep*10):.0f} Hz (每 10 物理步 = 1 控制步)")
print(f"\n   qpos 索引映射:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "(自由体)"
    addr = model.jnt_qposadr[i]
    print(f"     qpos[{addr:2d}] = {name}")

# ═══ 2. 实验：两步关节控制 ═══
print(f"\n{'='*55}")
print("实验：\"收拢姿态\" → \"展开姿态\" 两阶段关节控制")
print(f"{'='*55}")

# 姿态 A：机械臂"收拢"（关节角小）
pose_tucked = np.array([0.0, -0.3, -0.5, 0.0, 0.3, 0.0])
# 姿态 B：机械臂"展开"前伸（关节角大）
pose_reach = np.array([0.2, -1.2, -1.5, 0.0, -0.5, 0.0])

mujoco.mj_resetData(model, data)

# ── 阶段 1：移到收拢姿态 ──
print(f"\n  阶段 1: ctrl → {pose_tucked}")
data.ctrl[:] = pose_tucked

for step in range(200):
    mujoco.mj_step(model, data)
    if step % 50 == 0:
        err = np.abs(data.ctrl[:] - data.qpos[7:13]).max()
        print(f"    步{step:4d}: max|ctrl-qpos| = {err:.4f}")

print(f"    ✅ qpos = {np.round(data.qpos[7:13], 3)}")

# ── 阶段 2：移到展开姿态 ──
print(f"\n  阶段 2: ctrl → {pose_reach}")
data.ctrl[:] = pose_reach

for step in range(200):
    mujoco.mj_step(model, data)
    if step % 50 == 0:
        err = np.abs(data.ctrl[:] - data.qpos[7:13]).max()
        print(f"    步{step:4d}: max|ctrl-qpos| = {err:.4f}")

print(f"    ✅ qpos = {np.round(data.qpos[7:13], 3)}")

# ═══ 3. 总结 ═══
print(f"\n{'='*55}")
print("🔑 带走")
print(f"{'='*55}")
print(f"  mjModel    = 物理\"图纸\"（几何、质量、关节定义）— 只读")
print(f"  mjData     = 运行\"快照\"（位置 qpos、速度 qvel、接触力）")
print(f"  data.ctrl  = VLA 输出的目标 → PD 控制器跟踪")
print(f"  data.qpos  = 物理仿真后的实际关节角")
print(f"  mj_step()  = 推进 1 个物理步长（{model.opt.timestep}s）")
print(f"")
print(f"  📌 VLA 类比:")
print(f"     模型每 40ms 输出动作 → data.ctrl → 反复 mj_step → qpos 到位")
print(f"     200 步 × {model.opt.timestep}s = {200*model.opt.timestep:.1f}s 足以完成大范围关节移动")
