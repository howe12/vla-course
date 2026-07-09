#!/usr/bin/env python3
"""Step 3 评估实验：physical_success — 四阶段物理成功判断

判断一个 VLA 策略是否"真正完成了抓杯任务"，
不是只看末端位置（legacy_success），而是拆成四个阶段：
    ① 夹起 (Grasp) → ② 搬运 (Transport) → ③ 放置 (Place) → ④ 直立 (Upright)

用法：
    python3 codes/step3_eval/physical_success.py --demo      # 演示模式
    python3 codes/step3_eval/physical_success.py --compare   # 对比 legacy vs physical
"""

import sys
import os
import argparse
import math
import numpy as np

# 目标位置（与 sim_vla_arm.py 中一致）
DEFAULT_TARGET = np.array([0.35, 0.10, 0.05])


def check_physical_success(episode_data, target_pos, dt=0.02):
    """判断一个 episode 是否物理成功

    四阶段流水线——每个阶段依赖前一个阶段的结果。
    任何一个阶段失败，整个 episode 即为 physical_success = False。

    参数:
        episode_data: list[dict] — 每帧的数据
            每帧 dict 包含:
                "ee_pos":       (3,) ndarray — 末端执行器世界坐标
                "gripper_state": float       — 夹爪开度（0=闭合，>0=张开）
                "cup_pos":      (3,) ndarray — 杯子世界坐标
                "cup_orientation": (3,3) ndarray — 杯子旋转矩阵
        target_pos:   (3,) ndarray — 目标位置
        dt:           float — 帧间隔 (s)，默认 0.02

    返回:
        success: bool — 四阶段是否全部通过
        stages:  dict — {"grasp": bool, "transport": bool, "place": bool, "upright": bool}
        details: dict — 每阶段的关键帧号和判断值（用于调试）
    """
    stages = {"grasp": False, "transport": False, "place": False, "upright": False}
    details = {}

    # ===== 阶段 ①：夹起 (Grasp) =====
    # 条件：杯子 Z > 桌面 + 3cm（约 0.08m），且夹爪闭合（开度 < 1mm）
    grasp_frame = None
    for i, frame in enumerate(episode_data):
        cup_z = frame["cup_pos"][2]
        gripper_closed = frame["gripper_state"] < 0.001  # 1mm 以下视为闭合

        if cup_z > 0.08 and gripper_closed:
            stages["grasp"] = True
            grasp_frame = i
            details["grasp_frame"] = i
            details["grasp_cup_z"] = cup_z
            details["grasp_gripper"] = frame["gripper_state"]
            break

    if not stages["grasp"]:
        return False, stages, details

    # ===== 阶段 ②：搬运 (Transport) =====
    # 条件：从夹起到放置之间，杯子 Z 始终 > 7cm（未掉落）
    transport_ok = True
    place_frame = None

    for i in range(grasp_frame, len(episode_data)):
        frame = episode_data[i]

        # 检测掉落：杯子 Z 低于阈值
        if frame["cup_pos"][2] < 0.07:
            transport_ok = False
            details["transport_fail_frame"] = i
            details["transport_fail_cup_z"] = frame["cup_pos"][2]
            break

        # 检测放置：杯子靠近目标（XY 距离 < 5cm）且 Z 降到桌面附近
        dist_to_target = np.linalg.norm(
            frame["cup_pos"][:2] - target_pos[:2]
        )
        if dist_to_target < 0.05 and frame["cup_pos"][2] < 0.06:
            place_frame = i
            details["place_frame"] = i
            details["place_dist"] = dist_to_target
            details["place_cup_z"] = frame["cup_pos"][2]
            break

    if transport_ok and place_frame is not None:
        stages["transport"] = True
        stages["place"] = True  # 阶段 ③：放置
    else:
        stages["transport"] = transport_ok
        return False, stages, details

    # ===== 阶段 ④：直立 (Upright) =====
    # 条件：放置后杯子的上方向量与重力方向（0,0,1）夹角 < 15°
    cup_orientation = episode_data[place_frame]["cup_orientation"]
    # 杯子的 Z 轴方向（第三列）
    cup_up = cup_orientation[:, 2]
    z_axis = np.array([0.0, 0.0, 1.0])
    angle = np.arccos(np.clip(np.dot(cup_up, z_axis), -1.0, 1.0))
    angle_deg = np.degrees(angle)

    stages["upright"] = angle_deg < 15
    details["upright_angle"] = angle_deg

    success = all(stages.values())
    return success, stages, details


def evaluate_dataset(episodes, target_pos):
    """批量评估，输出四阶段统计

    参数:
        episodes: list[list[dict]] — 多个 episode 的数据
        target_pos: (3,) ndarray — 目标位置

    返回:
        report: dict — 总体统计
        results: list[dict] — 每个 episode 的详细结果
    """
    results = []
    for ep in episodes:
        success, stages, details = check_physical_success(ep, target_pos)
        results.append({
            "episode_id": ep[0].get("episode_id", "?"),
            "success": success,
            "stages": stages,
            "details": details,
        })

    total = len(results)
    if total == 0:
        return {"total_episodes": 0}, []

    report = {
        "total_episodes": total,
        "physical_success": sum(r["success"] for r in results),
        "physical_success_rate": sum(r["success"] for r in results) / total,
        "stage_grasp_rate":    sum(r["stages"]["grasp"]    for r in results) / total,
        "stage_transport_rate": sum(r["stages"]["transport"] for r in results) / total,
        "stage_place_rate":    sum(r["stages"]["place"]    for r in results) / total,
        "stage_upright_rate":  sum(r["stages"]["upright"]  for r in results) / total,
    }
    return report, results


def compare_success(episodes, target_pos):
    """对比 legacy_success 和 physical_success

    legacy_success 判定：末端执行器距目标 < 10cm 且 Z < 10cm（靠近桌面）

    返回:
        dict — legacy_rate, physical_rate, false_positive_rate
    """
    legacy_ok = 0
    physical_ok = 0
    false_positives = 0  # legacy=OK 但 physical=FAIL

    for ep in episodes:
        last_frame = ep[-1]
        last_ee = last_frame["ee_pos"]

        # legacy 判断：EE 在目标 10cm 内，且高度接近桌面
        legacy_dist = np.linalg.norm(last_ee - target_pos)
        legacy_success = legacy_dist < 0.10 and last_ee[2] < 0.10

        physical_success, _, _ = check_physical_success(ep, target_pos)

        if legacy_success:
            legacy_ok += 1
        if physical_success:
            physical_ok += 1
        if legacy_success and not physical_success:
            false_positives += 1

    total = len(episodes)
    return {
        "total_episodes": total,
        "legacy_ok": legacy_ok,
        "legacy_rate": legacy_ok / total if total > 0 else 0,
        "physical_ok": physical_ok,
        "physical_rate": physical_ok / total if total > 0 else 0,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / total if total > 0 else 0,
    }


# ===== 演示与测试 =====

def generate_demo_data():
    """生成 4 个典型 episode 的模拟数据用于演示"""
    episodes = []

    # Episode 1：假阳性 — EE 靠近了但没夹起来
    ep1 = []
    for i in range(200):
        t = i * 0.02
        progress = min(t / 3.0, 1)
        ep1.append({
            "episode_id": "ep01_false_positive",
            "ee_pos": np.array([0.35 * progress, 0.10 * progress, 0.10 - 0.05 * progress]),
            "gripper_state": 0.02,  # 始终张开！
            "cup_pos": np.array([0.35, 0.10, 0.05]),  # 杯子没动过
            "cup_orientation": np.eye(3),
        })
    episodes.append(ep1)

    # Episode 2：假阳性 — 搬运途中杯子掉落
    ep2 = []
    for i in range(200):
        t = i * 0.02
        progress = min(t / 3.0, 1)
        cup_z = 0.10 if i < 100 else 0.03  # 100 帧后掉落
        ep2.append({
            "episode_id": "ep02_dropped",
            "ee_pos": np.array([0.35 * progress, 0.10 * progress, 0.05]),
            "gripper_state": 0.0,
            "cup_pos": np.array([0.35 * progress, 0.10 * progress, cup_z]),
            "cup_orientation": np.eye(3),
        })
    episodes.append(ep2)

    # Episode 3：真阳性 — 完美完成
    ep3 = []
    for i in range(200):
        t = i * 0.02
        progress = min(t / 3.0, 1)
        cup_z = max(0.10, 0.10 - 0.05 * (i - 80) / 40) if i > 80 else 0.10
        if i > 120:
            cup_z = 0.055
        ep3.append({
            "episode_id": "ep03_true_positive",
            "ee_pos": np.array([0.35 * progress, 0.10 * progress, 0.05 + 0.05 * (1 - progress)]),
            "gripper_state": 0.0 if i > 40 else 0.02,
            "cup_pos": np.array([0.35 * progress, 0.10 * progress, max(0.055, cup_z)]),
            "cup_orientation": np.eye(3),
        })
    episodes.append(ep3)

    # Episode 4：假阳性 — 放置时杯子倒了
    ep4 = []
    for i in range(200):
        t = i * 0.02
        progress = min(t / 3.0, 1)
        cup_z = 0.055 if i > 120 else 0.10
        # 杯子倾倒：绕 X 轴旋转 30°
        angle = 0 if i < 120 else math.radians(30)
        rot = np.array([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ])
        ep4.append({
            "episode_id": "ep04_tipped_over",
            "ee_pos": np.array([0.35 * progress, 0.10 * progress, 0.05]),
            "gripper_state": 0.0 if i > 40 else 0.02,
            "cup_pos": np.array([0.35 * progress, 0.10 * progress, cup_z]),
            "cup_orientation": rot,
        })
    episodes.append(ep4)

    return episodes


def run_demo():
    """演示模式：跑 4 个典型 episode 并输出诊断"""
    print("=" * 60)
    print("Step 3 评估实验：physical_success 演示")
    print("=" * 60)
    print()

    episodes = generate_demo_data()
    target = DEFAULT_TARGET

    for ep in episodes:
        ep_id = ep[0]["episode_id"]
        success, stages, details = check_physical_success(ep, target)
        status = "✅ PASS" if success else "❌ FAIL"

        print(f"Episode: {ep_id}")
        print(f"  Result: {status}")
        print(f"  Stages: grasp={stages['grasp']}, transport={stages['transport']}, "
              f"place={stages['place']}, upright={stages['upright']}")
        if not success:
            failed_stages = [k for k, v in stages.items() if not v]
            print(f"  Failed at: {', '.join(failed_stages)}")
        if details:
            print(f"  Details: {details}")
        print()

    # 对比
    comparison = compare_success(episodes, target)
    print("=" * 60)
    print("legacy vs physical 对比")
    print("=" * 60)
    print(f"  Total episodes:     {comparison['total_episodes']}")
    print(f"  legacy_success:     {comparison['legacy_ok']}/{comparison['total_episodes']} "
          f"({comparison['legacy_rate']:.0%})")
    print(f"  physical_success:   {comparison['physical_ok']}/{comparison['total_episodes']} "
          f"({comparison['physical_rate']:.0%})")
    print(f"  False positives:    {comparison['false_positives']} "
          f"({comparison['false_positive_rate']:.0%})")
    print()
    if comparison['false_positive_rate'] > 0.2:
        print("⚠️  假阳性率 > 20%！legacy_success 非常不可靠。")
    else:
        print("✅ 假阳性率在可接受范围内。")


def run_compare():
    """对比模式：用 demo 数据跑 legacy vs physical 对比"""
    episodes = generate_demo_data()
    target = DEFAULT_TARGET

    comparison = compare_success(episodes, target)

    print("legacy vs physical_success 对比")
    print(f"  legacy_success:   {comparison['legacy_rate']:.0%}")
    print(f"  physical_success: {comparison['physical_rate']:.0%}")
    print(f"  False positive rate: {comparison['false_positive_rate']:.0%}")
    print()

    # 逐 episode 分析
    report, results = evaluate_dataset(episodes, target)
    print(f"四阶段通过率：")
    print(f"  grasp:     {report['stage_grasp_rate']:.0%}")
    print(f"  transport: {report['stage_transport_rate']:.0%}")
    print(f"  place:     {report['stage_place_rate']:.0%}")
    print(f"  upright:   {report['stage_upright_rate']:.0%}")


def main():
    parser = argparse.ArgumentParser(description="physical_success 评估脚本")
    parser.add_argument("--demo", action="store_true",
                        help="演示模式：跑 4 个典型 episode")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式：legacy vs physical")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.compare:
        run_compare()
    else:
        print("用法: python3 physical_success.py --demo | --compare")
        print()
        print("  --demo     演示模式，跑 4 个典型 episode 并输出诊断")
        print("  --compare  对比 legacy_success 和 physical_success 的差异")


if __name__ == "__main__":
    main()
