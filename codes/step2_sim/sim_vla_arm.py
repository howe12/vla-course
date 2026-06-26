#!/usr/bin/env python3
"""Step 2 仿真实验：MuJoCo 机械臂 + VLA 控制闭环

用一个简单的 6-DOF 机械臂模型，模拟 VLA 控制循环：
1. 加载 MuJoCo 模型（widowx_arm.xml）
2. 模拟"VLA 推理"（随机目标位置 → 关节角）
3. 在 Viewer 中实时观看机械臂移动

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
    """加载 6-DOF 机械臂 MuJoCo 模型"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    return model, data


def get_end_effector_pos(data):
    """获取末端执行器在世界坐标系中的位置"""
    site_id = mujoco.mj_name2id(
        data.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
    )
    return data.site_xpos[site_id].copy()


def vla_predict(data, step_count):
    """模拟 VLA 推理：根据当前状态输出目标关节角

    在实际课程中，这里会被替换为 OpenVLA/SmolVLA 模型推理。
    目前用简单的目标追踪逻辑模拟"VLA 理解了任务指令后的动作序列"。
    """
    # 模拟任务："把末端移到目标方块附近"
    target = np.array([0.35, 0.1, 0.08])  # 目标方块位置
    ee = get_end_effector_pos(data)
    dist = np.linalg.norm(ee - target)

    # 用简单的位置伺服：让末端逐渐靠近目标
    t = step_count * 0.02  # 20 Hz

    # 机械臂建在原点 (0,0,0.05)，需要大的关节运动才能到达目标
    # joint1: 基座旋转 → 指向目标方向
    # joint2: 肩部 → 向前伸
    # joint3: 肘部 → 弯向目标高度
    q = np.zeros(6)

    # 目标在 (0.35, 0.1, 0.08)，机械臂需要前伸+右转
    q[0] = math.atan2(target[1], target[0]) * (1 - math.exp(-t * 2))  # 逐渐转向目标
    q[1] = -1.2 + 0.3 * math.sin(t * 0.2)  # 肩部下压前伸
    q[2] = -1.5 + 0.4 * math.sin(t * 0.3)  # 肘部弯折
    q[3] = 0.0  # 腕稳定
    q[4] = -0.5  # 末端下指
    q[5] = 0.0

    return q, dist


def run_headless(model, data, steps=300):
    """无头模式：离屏渲染，保存帧供后续生成视频"""
    print("🎬 无头模式：渲染仿真帧...")

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []

    for step in range(steps):
        q_target, dist = vla_predict(data, step)

        # PD 控制：平滑追踪目标关节角
        for i in range(6):
            data.ctrl[i] = q_target[i]

        mujoco.mj_step(model, data)

        if step % 10 == 0:
            renderer.update_scene(data, camera="wrist_cam")
            frame = renderer.render()
            frames.append(frame.copy())

            ee = get_end_effector_pos(data)
            print(
                f"  Step {step:3d} | "
                f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) | "
                f"dist={dist:.3f}m"
            )

    print(f"\n渲染完成！共 {len(frames)} 帧")
    print(f"提示: 在有图形环境的机器上运行 'uv run python codes/step2_sim/sim_vla_arm.py' 查看实时动画")
    return frames


def run_viewer(model, data, steps=1000):
    """图形模式：在 MuJoCo Viewer 中实时观看"""
    print("🖥️  启动 MuJoCo Viewer...")
    print("   鼠标拖拽旋转视角 | 滚轮缩放 | 右键平移")
    print("   按 ESC 退出\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 重置到初始姿态
        mujoco.mj_resetData(model, data)

        step = 0
        while viewer.is_running():
            q_target, dist = vla_predict(data, step)

            # PD 控制
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
    print("│  当前用正弦轨迹模拟 VLA 输出。               │")
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
