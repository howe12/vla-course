#!/usr/bin/env python3
"""Step 2 仿真实验：MuJoCo 机械臂 + VLA 控制闭环

用一个简单的 6-DOF 机械臂模型，模拟 VLA 控制循环：
1. 加载 MuJoCo 模型（widowx_arm.xml）
2. 模拟"VLA 推理"（目标位置 → 关节角）
3. 在 Viewer 中实时观看机械臂移动，或离屏渲染

用法（需要图形环境）：
    uv run python codes/step2_sim/sim_vla_arm.py

用法（无头模式，保存渲染帧）：
    uv run python codes/step2_sim/sim_vla_arm.py --headless

按 ESC 退出 Viewer。
"""

import sys
import os
import time
import argparse
import math
import numpy as np

try:
    import mujoco
except ImportError:
    print("❌ MuJoCo 未安装")
    print("请运行: pip install mujoco")
    sys.exit(1)


# 模型文件路径（相对于项目根目录）
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "widowx_arm.xml",
)


def load_model():
    """加载 6-DOF 机械臂 MuJoCo 模型

    返回:
        model: MjModel — 模型的"蓝图"（几何、关节、物理参数）
        data:  MjData  — 模型的"快照"（当前时刻的位置、速度、力）

    关键概念：Model 在仿真全程不变（像建筑蓝图），
             Data 每步都在变（像建筑物的实时监控画面）。
    """
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    return model, data


def get_end_effector_pos(data):
    """获取末端执行器在世界坐标系中的位置

    MuJoCo 中用 site（位点）标记关键位置。
    end_effector 是 widowx_arm.xml 中定义的 site，
    位于夹爪尖端，用来追踪"手在哪里"。

    返回:
        ndarray(3,): 末端执行器的 (x, y, z) 世界坐标

    为什么用 .copy()？防止意外修改 MuJoCo 内部数据。
    """
    site_id = mujoco.mj_name2id(
        data.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
    )
    return data.site_xpos[site_id].copy()


def vla_predict(data, step_count):
    """模拟 VLA 推理：根据当前状态输出目标关节角

    ⚠️ 这是本课程的"VLA 替身"。
    在实际课程中，这里会被替换为 OpenVLA/SmolVLA 模型推理。
    目前用渐进轨迹模拟 VLA 理解了"把末端移到红色方块"后的动作序列。

    参数:
        data:       MjData 快照（用于获取当前末端位置）
        step_count: 当前步数（用于计算时间进度）

    返回:
        q:    6 个关节的目标角度 (ndarray, shape=(6,))
        dist: 末端到目标的欧氏距离 (float)

    关节约定：
        joint1: waist      — 绕 Z 轴，正角=逆时针，转向目标
        joint2: shoulder   — 绕 Y 轴，正角=向前（+X 方向）倾斜
        joint3: elbow      — 绕 Y 轴，正角=继续向前弯曲
        joint4: wrist_roll — 绕 Z 轴，末端自转
        joint5: wrist_pitch— 绕 Y 轴，末端俯仰
        joint6: wrist_yaw  — 绕 Z 轴，末端偏航
    """
    # 任务目标：红色方块在桌子上的位置
    target = np.array([0.35, 0.1, 0.08])
    ee = get_end_effector_pos(data)
    dist = np.linalg.norm(ee - target)

    t = step_count * 0.02       # 控制频率 20 Hz
    progress = min(t / 3.0, 1)  # 渐进因子 0→1，3 秒完成

    q = np.zeros(6)  # 6 个关节角

    # joint1: 基座旋转 — 转向目标方向（逆时针为正）
    q[0] = math.atan2(target[1], target[0]) * (1 - math.exp(-t * 1.5))

    # joint2: 肩部俯仰 — 正角向前（+X 方向）倾斜
    q[1] = 0.8 * progress

    # joint3: 肘部俯仰 — 正角继续向前弯曲，保持末端接近桌面高度
    q[2] = 0.6 * progress

    # joint4-6: 腕部 — 保持稳定，末端下指
    q[3] = 0.0
    q[4] = -0.3 * progress
    q[5] = 0.0

    return q, dist


def run_headless(model, data, steps=300):
    """无头模式：离屏渲染

    云容器没有显示器，必须用 OSMesa 软件渲染。
    每 10 步渲染一帧，共生成 steps/10 帧。

    这个循环就是 VLA 控制闭环的仿真版：
        观测 → 推理 → 执行 → 渲染
    Ch3-Ch6 只替换步骤 2（推理），其他三步不变。
    """
    print("🎬 无头模式：渲染仿真帧...")

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []

    for step in range(steps):
        q_target, dist = vla_predict(data, step)   # 1. VLA 推理 → 目标关节角

        # 2. PD 控制：将关节角平滑驱动到目标值
        for i in range(6):
            data.ctrl[i] = q_target[i]

        mujoco.mj_step(model, data)                 # 3. 物理仿真一步

        if step % 10 == 0:                          # 4. 每 10 步渲染一次
            renderer.update_scene(data)              #   更新场景
            frame = renderer.render()                #   渲染为像素数组
            frames.append(frame.copy())

            ee = get_end_effector_pos(data)
            print(
                f"  Step {step:3d} | "
                f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) | "
                f"dist={dist:.3f}m"
            )

    print(f"\n渲染完成！共 {len(frames)} 帧")
    return frames


def run_viewer(model, data, steps=1000):
    """图形模式：在 MuJoCo Viewer 中实时观看"""
    print("🖥️  启动 MuJoCo Viewer...")
    print("   鼠标拖拽旋转视角 | 滚轮缩放 | 右键平移")
    print("   按 ESC 退出\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)

        step = 0
        while viewer.is_running():
            q_target, dist = vla_predict(data, step)

            for i in range(6):
                data.ctrl[i] = q_target[i]

            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1

            if step % 50 == 0:
                ee = get_end_effector_pos(data)
                print(
                    f"  Step {step:3d} | "
                    f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) | "
                    f"dist_to_target={dist:.3f}m"
                )

            if step >= steps:
                break

    print(f"\n仿真结束！共 {step} 步")


def main():
    parser = argparse.ArgumentParser(description="VLA 机械臂仿真实验")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式（不弹窗口，保存渲染帧）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="仿真步数（默认 300）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2 仿真实验：MuJoCo 机械臂 + VLA 控制")
    print("=" * 60)
    print(f"\n模型: {MODEL_PATH}")
    print(f"关节数: 6-DOF（基座旋转 + 肩/肘 + 腕 x3）\n")

    model, data = load_model()

    print(f"物理步长: {model.opt.timestep}s ({1/model.opt.timestep:.0f} Hz)")
    print(f"控制维度: {model.nu}（6 个关节位置）")
    print(f"状态维度: nq={model.nq}, nv={model.nv}\n")

    # 仿真循环说明
    print("┌─────────────────────────────────────────────┐")
    print("│  这是 VLA 控制循环的仿真版：                 │")
    print("│                                              │")
    print("│  相机图像 ──→ [VLA 模型] ──→ 目标关节角      │")
    print("│     ↑                            ↓           │")
    print("│     └──── 末端位置 ←── [MuJoCo PD 控制]      │")
    print("│                                              │")
    print("│  当前用渐进式轨迹模拟 VLA 输出。              │")
    print("│  第 3 章的 OpenVLA 推理会替换这个 mock。      │")
    print("└─────────────────────────────────────────────┘\n")

    if args.headless:
        run_headless(model, data, args.steps)
    else:
        try:
            run_viewer(model, data, args.steps)
        except Exception as e:
            print(f"\n⚠️  Viewer 启动失败: {e}")
            print("可能是无图形环境（SSH），尝试无头模式:")
            print("  uv run python codes/step2_sim/sim_vla_arm.py --headless")
            sys.exit(1)

    print("\n⏭️  下一步: Step 3 — 用 OpenVLA 替换模拟的 VLA 推理")


if __name__ == "__main__":
    main()
