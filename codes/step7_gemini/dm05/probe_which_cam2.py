#!/usr/bin/env python3
"""对照实验：分别单独张开 left_gripper / right_gripper，比较两路腕部相机的变化。
   若「张左→A变化大、张右→B变化大」，则 A/B 分别装在该臂上（自视，可信）。
   若两次都是同一路变化大，说明那是「拍到对方」，需要用别的判据。"""
import os, sys, time
sys.path.insert(0, ".")
import cv2
import motor_executor_live_dm05 as m

OUT = "/home/leo/vla-dm05/whichcam2"; os.makedirs(OUT, exist_ok=True)
JN = m.JOINT_NAMES

def shot(tag, cmap):
    for idx, name in cmap:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(10): cap.grab()
        ok, fr = cap.retrieve()
        if ok: cv2.imwrite(f"{OUT}/{tag}_{name}_video{idx}.jpg", fr)
        cap.release()

def set_grip(live, idx, target, timeout=35.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = live.read_state_deg()
        if abs(target - cur[idx]) <= 1.5: break
        action = {f"{n}.pos": float(cur[i]) for i, n in enumerate(JN)}
        d = target - cur[idx]
        action[f"{JN[idx]}.pos"] = float(cur[idx] + (1.0 if d>0 else -1.0)*min(abs(d), 4.0))
        live.robot.send_action(action); time.sleep(0.6)
    return live.read_state_deg()

cmap = m.resolve_camera_map()
print("映射:", cmap)
live = m.LiveMotorController(calibrate=False)
try:
    print("[归位]"); live.home_all(step_deg=5.0, tol_deg=1.5, timeout_s=180.0, settle_s=1.2)
    cur = live.read_state_deg()
    print(f"  L_grip={cur[6]:.1f} R_grip={cur[13]:.1f}")

    print("\n[A] 基准（双爪都 ~15）"); shot("base", cmap)

    print("[B] 只张 left_gripper → 60")
    cur = set_grip(live, 6, 60.0); print(f"  L_grip={cur[6]:.1f} R_grip={cur[13]:.1f}")
    shot("leftopen", cmap)

    print("[C] 复位 left，只张 right_gripper → 60")
    set_grip(live, 6, 15.0); cur = set_grip(live, 13, 60.0)
    print(f"  L_grip={cur[6]:.1f} R_grip={cur[13]:.1f}")
    shot("rightopen", cmap)

    print("[D] 复位")
    set_grip(live, 13, 15.0)
finally:
    live.close()
print("完成")
