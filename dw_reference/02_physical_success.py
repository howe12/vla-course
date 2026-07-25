#!/usr/bin/env python3
"""Task 02: 物理成功评估 — legacy_success vs physical_success

来源: Datawhale 02_physical_success_review.ipynb
用法: python3 dw_reference/02_physical_success.py
前置: 无（纯 Python，不需要 GPU）
"""

import json
from pathlib import Path

ASSET_DIR = Path(__file__).parent / "assets"

def physical_success(row, min_lift=0.03, min_lift_steps=3, upright_threshold=0.7):
    """四阶段物理成功判定 (AND 逻辑)"""
    legacy = bool(row.get("legacy_success", row.get("success", False)))
    max_lift = float(row.get("max_target_lift", row.get("max_mug_lift", 0.0)))
    lifted_steps = int(row.get("lifted_steps", 0))
    upright = float(row.get("final_target_upright_cos", row.get("final_mug_upright_cos", 1.0)))

    # ① 夹起: 杯子离桌 >= 3cm 且持续 >= 3 步
    grasp = max_lift >= min_lift and lifted_steps >= min_lift_steps
    # ②+③ 搬运+放置: 由 legacy 判定（此处简化，实际需更多字段）
    transport_place = legacy
    # ④ 直立: 杯子方向与重力夹角 < arccos(0.7) ≈ 45°
    upright_ok = upright >= upright_threshold

    return {
        "legacy": legacy,
        "grasp": grasp,
        "transport_place": transport_place,
        "upright": upright_ok,
        "physical": grasp and transport_place and upright_ok,
    }

def main():
    print("=" * 60)
    print("Task 02: 物理成功评估")
    print("=" * 60)

    # --- 展示判定逻辑 ---
    print("\n>>> 四阶段判定逻辑\n")
    print("① 夹起 (Grasp)    : 杯子离桌 >= 3cm，持续 >= 3 步")
    print("② 搬运 (Transport) : 杯中不掉落（Z 坐标始终 > 2cm）")
    print("③ 放置 (Place)     : 杯子 XY 距目标 < 5cm")
    print("④ 直立 (Upright)   : 杯子方向与重力夹角 < 45°")
    print("\n四阶段为 AND 关系——任一失败则整体 physical_success = False\n")

    # --- 三个案例 ---
    print(">>> 案例演示\n")

    examples = [
        {
            "name": "Case 1 — 真阳性",
            "data": {"success": True, "max_mug_lift": 0.09, "lifted_steps": 120, "final_mug_upright_cos": 0.96},
            "desc": "四阶段全部通过 → physical_success = True"
        },
        {
            "name": "Case 2 — 假阳性（夹爪没闭合）",
            "data": {"success": True, "max_mug_lift": 0.005, "lifted_steps": 0, "final_mug_upright_cos": 0.99},
            "desc": "legacy 说成功（EE 位置到了），但杯子没动 → physical_success = False"
        },
        {
            "name": "Case 3 — 临界（杯子倒了）",
            "data": {"success": True, "max_mug_lift": 0.08, "lifted_steps": 80, "final_mug_upright_cos": 0.2},
            "desc": "前三阶段通过但杯子倾斜 > 45° → physical_success = False"
        },
    ]

    for ex in examples:
        result = physical_success(ex["data"])
        status = "✅" if result["physical"] else "❌"
        print(f"  {ex['name']}")
        print(f"    legacy:   {'✅' if result['legacy'] else '❌'}")
        print(f"    grasp:    {'✅' if result['grasp'] else '❌'}")
        print(f"    upright:  {'✅' if result['upright'] else '❌'}")
        print(f"    physical: {status}")
        print(f"    → {ex['desc']}")
        print()

    # --- 读取 Datawhale 指标快照 ---
    snapshot_path = ASSET_DIR / "metrics_snapshot.json"
    if snapshot_path.exists():
        print(">>> Datawhale 实测数据\n")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if "best_summary" in snapshot:
            for k, v in snapshot["best_summary"].items():
                print(f"  {k}: {v}")
    else:
        print(f"  (指标快照未找到: {snapshot_path})")
        print(f"  请从 Datawhale 仓库复制 assets/metrics_snapshot.json 到此目录")

    print("\n✅ Task 02 完成。理解 legacy vs physical 后，继续 Task 03 ACT 诊断。")

if __name__ == "__main__":
    main()
