#!/usr/bin/env python3
"""Step 3 实验：评估真实 episode — 用 physical_success 评测采集的数据

读取 record_episode.py 采集的 episode 文件，
运行 physical_success 四阶段评估，对比 legacy_success。

用法：
    python3 step3_eval/evaluate_real.py --data data/episodes/
"""

import sys
import os
import argparse
import glob
import json
import numpy as np

# 导入 physical_success 核心函数
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step3_eval.physical_success import (
    check_physical_success,
    evaluate_dataset,
    compare_success as compare_legacy_physical,
)

# 目标位置
TARGET_POS = np.array([0.35, 0.10, 0.05])


def load_episodes(data_dir):
    """从 .npz 文件加载 episode 数据，转换为 physical_success.py 所需格式

    返回:
        episodes: list[list[dict]] — 每个 episode 是一个帧列表
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))

    if not npz_files:
        print(f"❌ 在 {data_dir}/ 中未找到 episode_*.npz 文件")
        sys.exit(1)

    episodes = []
    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)

        ee_pos = data["ee_pos"]          # (N, 3)
        target_pos = data["target_pos"]  # (N, 3)
        target_orient = data["target_orientation"]  # (N, 3, 3)
        gripper_dist = data["gripper_dist"]  # (N,)

        ep_id = os.path.basename(fpath).replace(".npz", "")
        n_frames = len(ee_pos)

        frames = []
        for i in range(n_frames):
            # 模拟夹爪状态：EE 距目标 < 3cm 视为"夹住了"
            gripper_state = 0.0 if gripper_dist[i] < 0.03 else 0.05

            frames.append({
                "episode_id": ep_id,
                "frame_index": i,
                "ee_pos": ee_pos[i],
                "gripper_state": gripper_state,
                "cup_pos": target_pos[i],           # 目标方块 = "杯子"
                "cup_orientation": target_orient[i],  # 3x3 旋转矩阵
            })

        episodes.append(frames)

    return episodes


def print_episode_detail(episode, target_pos):
    """打印单个 episode 的详细评估结果"""
    ep_id = episode[0]["episode_id"]
    success, stages, details = check_physical_success(episode, target_pos)

    status = "✅ PASS" if success else "❌ FAIL"
    failed = [k for k, v in stages.items() if not v]

    print(f"\n{'='*60}")
    print(f"Episode: {ep_id}")
    print(f"{'='*60}")
    print(f"  结果: {status}")
    print(f"  四阶段: grasp={stages['grasp']}, transport={stages['transport']}, "
          f"place={stages['place']}, upright={stages['upright']}")

    if failed:
        print(f"  失败阶段: {', '.join(failed)}")
        for stage_name in failed:
            if stage_name == "grasp":
                # 检查为什么没夹起来
                min_dist = min(f["gripper_dist"] for f in episode)
                gripper_states = set(f["gripper_state"] for f in episode)
                print(f"    → grasp 失败原因: EE 距目标最小={min_dist*100:.1f}cm, "
                      f"夹爪状态={gripper_states}")
            elif stage_name == "transport":
                # 找到最低的杯子 Z
                min_z = min(f["cup_pos"][2] for f in episode)
                print(f"    → transport 失败原因: 杯子最低 Z={min_z:.3f}m (阈值 0.07m)")
            elif stage_name == "place":
                last_cup = episode[-1]["cup_pos"]
                dist = np.linalg.norm(last_cup[:2] - target_pos[:2])
                print(f"    → place 失败原因: 最终 XY 距离={dist*100:.1f}cm (阈值 5cm)")
            elif stage_name == "upright":
                print(f"    → upright 失败: 杯子倾斜角={details.get('upright_angle', '?')}°")

    # 最后一帧信息
    last = episode[-1]
    print(f"  最后一帧: EE=({last['ee_pos'][0]:.3f},{last['ee_pos'][1]:.3f},"
          f"{last['ee_pos'][2]:.3f})")


def main():
    parser = argparse.ArgumentParser(description="评估采集的 episode")
    parser.add_argument("--data", type=str, default="data/episodes",
                        help="episode 数据目录")
    parser.add_argument("--detail", action="store_true",
                        help="逐 episode 打印详细信息")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 3 实验：评估真实 episode")
    print("=" * 60)
    print()

    episodes = load_episodes(args.data)
    print(f"加载 {len(episodes)} 个 episode（来自 {args.data}/）\n")

    # 批量评估
    report, results = evaluate_dataset(episodes, TARGET_POS)

    print("📊 四阶段评估报告")
    print(f"  physical_success:     {report['physical_success']}/{report['total_episodes']} "
          f"({report['physical_success_rate']:.0%})")
    print(f"  grasp 通过率:         {report['stage_grasp_rate']:.0%}")
    print(f"  transport 通过率:     {report['stage_transport_rate']:.0%}")
    print(f"  place 通过率:         {report['stage_place_rate']:.0%}")
    print(f"  upright 通过率:       {report['stage_upright_rate']:.0%}")
    print()

    # 与 legacy 对比
    comparison = compare_legacy_physical(episodes, TARGET_POS)
    print("📊 legacy vs physical 对比")
    print(f"  legacy_success:       {comparison['legacy_ok']}/{comparison['total_episodes']} "
          f"({comparison['legacy_rate']:.0%})")
    print(f"  physical_success:     {comparison['physical_ok']}/{comparison['total_episodes']} "
          f"({comparison['physical_rate']:.0%})")
    print(f"  假阳性 (legacy=OK, physical=FAIL): {comparison['false_positives']} "
          f"({comparison['false_positive_rate']:.0%})")
    print()

    if args.detail:
        for ep in episodes:
            print_episode_detail(ep, TARGET_POS)

    # 诊断建议
    if comparison['false_positive_rate'] > 0.2:
        print("⚠️  假阳性率 > 20%！")
        print("   → legacy_success 严重高估了策略性能")
        print("   → 建议：用 physical_success 替代 legacy_success 作为主要评估指标")
    else:
        print("✅ 假阳性率在可接受范围内")


if __name__ == "__main__":
    main()
