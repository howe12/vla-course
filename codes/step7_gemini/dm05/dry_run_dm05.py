#!/usr/bin/env python3
"""DM0.5 DRY_RUN 全链路测试: 相机 → HTTP推理 → motor_executor(只记录不驱动)

用法: python dry_run_dm05.py [--rounds N] [--addr URL]
安全: 全程 DRY_RUN，不驱动电机，只打印执行日志
"""
import argparse
import time
import sys
import os
import logging
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain_http_client import DM05HTTPClient, CHUNK_SIZE, ACTION_DIM
from motor_executor import MotorExecutor, SafetyConfig, JOINT_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dm05_dryrun")

CAMERA_MAP = [(0, "front"), (4, "left"), (2, "right")]
INSTRUCTION = "grab two objects into the middle box"
STATE_DIM = 14
DEG_TO_0P1DEG = 10.0


def main():
    ap = argparse.ArgumentParser(description="DM0.5 DRY_RUN 全链路测试")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-joint-delta", type=float, default=20.0)
    ap.add_argument("--addr", default="http://127.0.0.1:7891")
    args = ap.parse_args()

    # 连接推理服务
    client = DM05HTTPClient(args.addr)
    if not client.ping():
        logger.error(f"无法连接 DM0.5: {args.addr}")
        logger.info("提示: 确保 SSH 隧道已建立 (tunnel_dm05.sh)")
        return
    logger.info(f"DM0.5 连接 OK: {args.addr}")

    executor = MotorExecutor(SafetyConfig(dry_run=True, max_joint_delta=args.max_joint_delta))

    # 开相机
    caps = {}
    for idx, name in CAMERA_MAP:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            caps[name] = cap
    logger.info(f"相机已开: {list(caps.keys())}")

    for round_i in range(args.rounds):
        print(f"\n{'='*50}\n=== Round {round_i+1}/{args.rounds} ===")

        # 1. 采集
        t_cam = time.time()
        images_jpeg = {}
        for name, cap in caps.items():
            for _ in range(2):
                cap.read()
            ok, frame = cap.read()
            if not ok:
                print(f"  !! 相机 {name} 采集失败")
                continue
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            images_jpeg[name] = buf.tobytes()
        cam_ms = (time.time() - t_cam) * 1000
        print(f"  [采集] {cam_ms:.0f}ms, {len(images_jpeg)} 路图像")

        # 2. HTTP 推理 (state 用零向量，DRY_RUN 不需要真实姿态)
        state_deg = [0.0] * STATE_DIM
        t_inf = time.time()
        actions, server_latency = client.predict(
            state_deg=state_deg,
            images_jpeg=images_jpeg,
            instruction=INSTRUCTION,
        )
        total_ms = (time.time() - t_inf) * 1000
        print(f"  [推理] 总 {total_ms:.0f}ms (云端 {server_latency:.0f}ms), chunk={actions.shape}")

        # 3. 动作统计
        print(f"  [动作] 范围 [{actions.min():.2f}, {actions.max():.2f}]°, "
              f"均值 {actions.mean():.2f}°")

        # 4. 逐步送 executor (DRY_RUN)
        for step in range(min(actions.shape[0], 10)):  # DRY_RUN 只跑前10步
            step_actions_0p1 = [v * DEG_TO_0P1DEG for v in actions[step].tolist()]
            executor.execute_action_chunk(step_actions_0p1, chunk_size=CHUNK_SIZE, action_dim=ACTION_DIM)
            time.sleep(0.02)

        # 5. 状态回读
        state = executor.get_state()
        smp = {k: round(v, 1) for k, v in list(state.items())[:5]}
        print(f"  [状态] 前5关节 (0.1°): {smp}")

    print(f"\n{'='*50}")
    print(f"DRY_RUN 完成，共 {executor.action_count} 步被校验（未驱动电机）")

    for cap in caps.values():
        cap.release()


if __name__ == "__main__":
    main()
