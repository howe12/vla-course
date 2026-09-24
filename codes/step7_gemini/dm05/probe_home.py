#!/usr/bin/env python3
"""独立归位探针：只做归位，不跑推理，用于验证 home_all 是否真能到位。

用法: python3 probe_home.py [step_deg]
"""
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe")

sys.path.insert(0, ".")
from motor_executor_live_dm05 import (  # noqa: E402
    LiveMotorController, TRAINING_INITIAL_POSE, JOINT_NAMES,
)

step = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

live = LiveMotorController(calibrate=False)
try:
    t0 = time.time()
    before = live.read_state_deg()
    logger.info("归位前: " + ", ".join(
        f"{n}={before[i]:.1f}" for i, n in enumerate(JOINT_NAMES)))
    after = live.home_all(step_deg=step, tol_deg=1.5, timeout_s=240.0, settle_s=1.2)
    dt = time.time() - t0

    print("\n" + "=" * 78)
    print(f"{'关节':<22}{'归位前':>10}{'目标':>10}{'归位后':>10}{'残余':>10}")
    print("-" * 78)
    worst = 0.0
    for i, n in enumerate(JOINT_NAMES):
        tgt = TRAINING_INITIAL_POSE[n]
        res = after[i] - tgt
        worst = max(worst, abs(res))
        flag = "  ✅" if abs(res) <= 1.5 else ("  ⚠️" if abs(res) <= 5 else "  ❌")
        print(f"{n:<22}{before[i]:>10.1f}{tgt:>10.1f}{after[i]:>10.1f}{res:>+10.1f}{flag}")
    print("-" * 78)
    print(f"最大残余偏差: {worst:.2f}°   总耗时: {dt:.1f}s")
    print("=" * 78)
finally:
    live.close()
