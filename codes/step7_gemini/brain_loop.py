#!/usr/bin/env python3
"""
Gemini 小脑 (Jetson Orin NX) — 大小脑通信核心循环

职责:
  1. 从相机 + 关节编码器读取观测
  2. 通过 HTTP POST 发送到云端大脑 (L40)
  3. 接收推理动作 → SGR 帧协议下发电机
  4. 100Hz 安全监控（与通信并行）

依赖: requests, numpy, opencv-python-headless, pyserial
环境: conda activate vla_brain
"""

import argparse, json, time, sys, os
import requests
import numpy as np
import cv2

# ── 配置 ──────────────────────────────────────────
OBS_FREQ = 15        # VLA 观测频率 (Hz)，与推理延迟匹配
CTRL_FREQ = 100      # 底层控制频率 (Hz)
SAFE_POSE = [0.0, 85.0, -87.5, 0.0, -11.6, 0.0, -80.0]  # 安全位姿 (°)

def get_observation(camera_ids=None):
    """
    从 Gemini 传感器读取观测 → JSON 序列化。

    Returns: dict {images: {"front": (224,224,3) list}, joint_state: [7]}

    TODO: 实际部署时替换为 real camera + SGR motor read
    """
    obs = {"images": {}, "joint_state": []}

    if camera_ids is None:
        camera_ids = [0]  # 默认用 /dev/video0
    for i, cam_id in enumerate(camera_ids):
        cap = cv2.VideoCapture(cam_id)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            obs["images"][f"cam_{i}"] = frame.tolist()
    # TODO: read joint_state from SGR motor (current_positions dict)
    obs["joint_state"] = [0.0] * 7

    return obs

def send_to_brain(brain_url, obs, instruction=""):
    """
    HTTP POST → 云端大脑，返回动作。

    API contract:
      POST /predict
      Body: {"images": {...}, "joint_state": [...], "instruction": "..."}
      Response: {"action": [Δx,Δy,Δz,Δrx,Δry,Δrz,gripper]}  # 7维增量
    """
    payload = {"images": obs["images"], "joint_state": obs["joint_state"],
               "instruction": instruction}
    try:
        resp = requests.post(f"{brain_url}/predict", json=payload, timeout=3)
        return resp.json().get("action", [0]*7)
    except Exception as e:
        print(f"[brain] unreachable: {e}")
        return None

def safety_clip(action, max_delta_xyz=0.05, max_delta_rot=0.26):
    """L1 安全：动作钳制"""
    action = np.array(action, dtype=np.float32)
    action[:3] = np.clip(action[:3], -max_delta_xyz, max_delta_xyz)
    action[3:6] = np.clip(action[3:6], -max_delta_rot, max_delta_rot)
    action[6] = np.clip(action[6], -1.0, 1.0)
    return action.tolist()

def execute_action(action):
    """
    下发动作到 SGR 电机。

    TODO: 实际部署时调用 sgr.sync_write(0x01, 0x0B, goal_positions)
    """
    print(f"  action: Δxyz=[{action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f}]  "
          f"Δrot=[{action[3]:+.3f},{action[4]:+.3f},{action[5]:+.3f}]  grip={action[6]:.0f}")

def emergency_stop():
    """急停：发送安全位姿 → 关力矩 → 退出"""
    print("[!] EMERGENCY STOP activated")
    # TODO: sgr.sync_write(0x01, 0x0B, [j*10 for j in SAFE_POSE])
    # import time; time.sleep(2)
    # TODO: sgr.sync_write(0x01, 0x08, 0)  # torque disable
    sys.exit(1)

# ── 主循环 ────────────────────────────────────────
def main(brain_url, instruction, max_steps):
    print(f"Gemini Brain Loop — cloud brain: {brain_url}")
    print(f"Task: {instruction or '(no instruction)'}")
    print(f"OBS={OBS_FREQ}Hz | CTRL={CTRL_FREQ}Hz | steps={max_steps}")

    for step in range(max_steps):
        t0 = time.time()

        # 1. 读观测
        obs = get_observation()
        if not obs["images"]:
            print(f"[{step}] no image — skip")
            time.sleep(0.05)
            continue

        # 2. 发送到大脑
        action = send_to_brain(brain_url, obs, instruction)
        if action is None:
            print(f"[{step}] brain timeout — holding position")
            emergency_stop()
            break

        # 3. 安全检查 + 执行
        action = safety_clip(action)
        execute_action(action)

        # 4. 维持观测频率
        elapsed = time.time() - t0
        sleep_t = max(0, 1.0/OBS_FREQ - elapsed)
        print(f"[{step}] done ({elapsed*1000:.0f}ms, slept {sleep_t*1000:.0f}ms)")
        time.sleep(sleep_t)

    print("Loop complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-url", default="http://localhost:8001",
                    help="云端大脑 API 地址")
    ap.add_argument("--instruction", default="",
                    help="自然语言任务指令")
    ap.add_argument("--max-steps", type=int, default=100,
                    help="最大推理步数")
    args = ap.parse_args()
    main(args.brain_url, args.instruction, args.max_steps)
