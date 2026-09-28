#!/usr/bin/env python3
"""物理判定相机归属：只把「left_*」索引的那条臂的夹爪张到最大，其余全部不动。
   然后抓 video2 / video4 的图，人眼一看便知哪路相机装在这条臂上。"""
import os, sys, time
sys.path.insert(0, ".")
import cv2
import motor_executor_live_dm05 as m

OUT = "/home/leo/vla-dm05/whichcam"
os.makedirs(OUT, exist_ok=True)

def shot(tag, cmap, n_discard=8):
    saved = []
    for idx, name in cmap:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(n_discard):
            cap.grab()
        ok, fr = cap.retrieve()
        if ok:
            fn = f"{OUT}/{tag}_{name}_video{idx}.jpg"
            cv2.imwrite(fn, fr)
            saved.append((name, idx, round(float(fr.mean()),1)))
        cap.release()
    return saved

cmap = m.resolve_camera_map()
print("相机映射:", cmap)
print("路径绑定:", m.CAMERA_USB_PATH)

live = m.LiveMotorController(calibrate=False)
try:
    # 1) 归位到训练初始位姿
    print("\n[1] 整机归位...")
    live.home_all(step_deg=5.0, tol_deg=1.5, timeout_s=180.0, settle_s=1.2)
    cur = live.read_state_deg()
    print(f"    归位后 L_grip={cur[6]:.1f}  R_grip={cur[13]:.1f}")

    # 2) 抓「动作前」基准图
    print("\n[2] 抓取动作前的基准图...")
    for nm, idx, mn in shot("before", cmap):
        print(f"    {nm:6s} video{idx}  mean={mn}")

    # 3) 只把「left」索引的夹爪（JOINT_NAMES[6] = left_gripper）张到 60
    JN = m.JOINT_NAMES
    target = 60.0
    print(f"\n[3] 只张开 left_gripper (index 6) → {target}°，right 保持不动...")
    t0 = time.time()
    while time.time() - t0 < 30:
        cur = live.read_state_deg()
        if target - cur[6] <= 1.5:
            break
        action = {f"{n}.pos": float(cur[i]) for i, n in enumerate(JN)}
        d = target - cur[6]
        action[f"{JN[6]}.pos"] = float(cur[6] + (1.0 if d > 0 else -1.0) * min(abs(d), 4.0))
        live.robot.send_action(action)
        time.sleep(0.6)
    cur = live.read_state_deg()
    print(f"    L_grip={cur[6]:.1f}  L_shldr={cur[1]:.1f}  R_grip={cur[13]:.1f}  R_shdr={cur[8]:.1f}")

    # 4) 抓「动作后」图
    print("\n[4] 抓取动作后的图...")
    for nm, idx, mn in shot("after", cmap):
        print(f"    {nm:6s} video{idx}  mean={mn}")
    print("\n请对比 before_*/after_* —— 夹爪张开的那个相机就是 left_* 那条臂的相机")
finally:
    live.close()
