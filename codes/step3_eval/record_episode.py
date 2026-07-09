#!/usr/bin/env python3
"""Step 3 实验：采集真实 episode — 从 MuJoCo 仿真记录完整数据

扩展 Ch2 的 sim_vla_arm.py，除了运行控制策略外，
额外记录每一帧的完整数据，供 physical_success 评估使用。

记录字段（每帧）：
    ee_pos:         末端执行器 (x, y, z)
    target_pos:     目标方块 (x, y, z)
    target_orient:  目标方块 3x3 旋转矩阵
    gripper_dist:   EE 到目标的距离（模拟夹爪"够到了没有"）
    frame_index:    帧序号

用法：
    # 采集 1 个 episode（默认 300 帧）
    MUJOCO_GL=osmesa python3 step3_eval/record_episode.py --episodes 5 --out data/
    
    # 用不同策略采集
    MUJOCO_GL=osmesa python3 step3_eval/record_episode.py --policy approach --episodes 3 --out data/
"""

import sys
import os
import argparse
import json
import math
import time
import numpy as np

try:
    import mujoco
except ImportError:
    print("❌ MuJoCo 未安装")
    sys.exit(1)


# 模型路径
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models", "widowx_arm.xml",
)


def load_model():
    """加载 6-DOF 机械臂 MuJoCo 模型"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        sys.exit(1)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    return model, data


def get_ee_pos(data):
    """末端执行器世界坐标"""
    site_id = mujoco.mj_name2id(
        data.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
    )
    return data.site_xpos[site_id].copy()


def get_target_pos(data):
    """目标方块（红色方块）的世界坐标"""
    body_id = mujoco.mj_name2id(
        data.model, mujoco.mjtObj.mjOBJ_BODY, "target"
    )
    return data.xpos[body_id].copy()


def get_target_orientation(data):
    """目标方块的 3x3 旋转矩阵"""
    body_id = mujoco.mj_name2id(
        data.model, mujoco.mjtObj.mjOBJ_BODY, "target"
    )
    return data.xmat[body_id].reshape(3, 3).copy()


def compute_control(step, total_steps, policy="approach"):
    """根据策略名计算 6 个关节目标角度

    策略说明：
        approach — 渐进式靠近目标（Ch2 修正版），机械臂从竖直→向前伸
        sine     — 正弦振荡策略（旧版），机械臂来回摆动
        hover    — 停在目标上方但不下降（故意不完成放置）

    返回:
        q: 6 个关节角
    """
    t = step * 0.02               # 20 Hz
    progress = min(t / 3.0, 1)    # 0→1

    q = np.zeros(6)
    target_xy = np.array([0.35, 0.10])

    if policy == "approach":
        # 渐进策略：从竖直位姿平滑伸向目标
        q[0] = math.atan2(target_xy[1], target_xy[0]) * (1 - math.exp(-t * 1.5))
        q[1] = 0.8 * progress
        q[2] = 0.6 * progress
        q[3] = 0.0
        q[4] = -0.3 * progress
        q[5] = 0.0

    elif policy == "sine":
        # 正弦振荡：机械臂来回摆动，可能"碰巧"经过目标但未真正抓取
        q[0] = math.atan2(target_xy[1], target_xy[0]) * (1 - math.exp(-t * 2))
        q[1] = -1.2 + 0.3 * math.sin(t * 0.2)
        q[2] = -1.5 + 0.4 * math.sin(t * 0.3)
        q[3] = 0.0
        q[4] = -0.5
        q[5] = 0.0

    elif policy == "hover":
        # 悬停策略：到达目标上方后不下沉（测试 place=False）
        q[0] = math.atan2(target_xy[1], target_xy[0]) * (1 - math.exp(-t * 1.5))
        q[1] = 0.5 * min(t / 1.5, 1)   # 只弯一半
        q[2] = 0.3 * min(t / 1.5, 1)   # 肘部也少弯
        q[3] = 0.0
        q[4] = 0.0                      # 不下指
        q[5] = 0.0

    return q


def run_episode(model, data, steps=300, policy="approach"):
    """运行一个 episode 并记录每帧数据

    返回:
        episode: list[dict] — 每帧的完整数据
    """
    # 重置仿真
    mujoco.mj_resetData(model, data)

    episode = []
    target_pos_target = np.array([0.35, 0.10, 0.05])

    for step in range(steps):
        # 1. 计算控制信号
        q_target = compute_control(step, steps, policy)

        # 2. 写入 PD 控制
        for i in range(6):
            data.ctrl[i] = q_target[i]

        # 3. 物理仿真
        mujoco.mj_step(model, data)

        # 4. 记录数据（每帧都记）
        ee = get_ee_pos(data)
        tpos = get_target_pos(data)
        torient = get_target_orientation(data)

        episode.append({
            "frame_index": step,
            "ee_pos": ee.tolist(),
            "target_pos": tpos.tolist(),
            "target_orientation": torient.tolist(),
            "gripper_dist": float(np.linalg.norm(ee - tpos)),
        })

    return episode


def record_episodes(num_episodes, policy, out_dir, steps=300):
    """采集多个 episode 并保存"""
    model, data = load_model()

    os.makedirs(out_dir, exist_ok=True)

    print(f"模型: {MODEL_PATH}")
    print(f"策略: {policy}")
    print(f"帧数/episode: {steps}")
    print(f"采集 {num_episodes} 个 episode...\n")

    for ep_idx in range(num_episodes):
        episode = run_episode(model, data, steps, policy)

        # 最后一帧的统计
        last = episode[-1]
        dist_to_target = np.linalg.norm(
            np.array(last["ee_pos"]) - np.array([0.35, 0.10, 0.05])
        )

        # 保存为 numpy 文件
        filename = os.path.join(out_dir, f"episode_{ep_idx:03d}.npz")
        # 构造成 physical_success.py 所需的格式
        np.savez(
            filename,
            ee_pos=np.array([f["ee_pos"] for f in episode]),
            target_pos=np.array([f["target_pos"] for f in episode]),
            target_orientation=np.array([f["target_orientation"] for f in episode]),
            gripper_dist=np.array([f["gripper_dist"] for f in episode]),
            policy=policy,
            steps=steps,
        )

        print(f"  Episode {ep_idx:03d}: {steps} 帧 "
              f"| 最终 EE→目标={dist_to_target:.3f}m "
              f"| 目标位置={last['target_pos']} "
              f"→ {filename}")

    print(f"\n✅ {num_episodes} 个 episode 已保存到 {out_dir}/")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="采集仿真 episode")
    parser.add_argument("--episodes", type=int, default=4,
                        help="采集数量")
    parser.add_argument("--policy", type=str, default="approach",
                        choices=["approach", "sine", "hover"],
                        help="控制策略")
    parser.add_argument("--steps", type=int, default=300,
                        help="每 episode 帧数")
    parser.add_argument("--out", type=str, default="data/episodes",
                        help="输出目录")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 3 实验：采集仿真 episode")
    print("=" * 60)
    print()

    record_episodes(args.episodes, args.policy, args.out, args.steps)


if __name__ == "__main__":
    main()
