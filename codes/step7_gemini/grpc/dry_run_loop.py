#!/usr/bin/env python3
"""DRY_RUN 全链路测试: 相机 → gRPC推理 → motor_executor(只记录不驱动)

用法: python dry_run_loop.py [--rounds N] [--verbose]
安全: 全程 DRY_RUN，不驱动电机，只打印执行日志
"""
import argparse
import time
import sys
import logging
import numpy as np
import cv2
import grpc

sys.path.insert(0, "/home/leo/vla-grpc/codes/step7_gemini/grpc")
import embodied_brain_pb2 as pb2
import embodied_brain_pb2_grpc as pb2_grpc
from motor_executor import MotorExecutor, SafetyConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dryrun")

# 相机映射（NUC 3路）
CAMERA_MAP = [(0, "front"), (2, "left"), (5, "right")]
INSTRUCTION = "grab two objects into the middle box"
GRPC_ADDR = "127.0.0.1:50051"
STATE_DIM = 14
CHUNK_SIZE = 5
ACTION_DIM = 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-joint-delta", type=float, default=20.0)
    args = ap.parse_args()

    channel = grpc.insecure_channel(GRPC_ADDR)
    stub = pb2_grpc.BrainServiceStub(channel)

    executor = MotorExecutor(SafetyConfig(dry_run=True, max_joint_delta=args.max_joint_delta))

    # 预开相机
    caps = {}
    for idx, name in CAMERA_MAP:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        caps[name] = cap
    print(f"[dryrun] 相机已开: {list(caps.keys())}")

    # Ping 确认连接
    try:
        r = stub.Ping(pb2.PingRequest(timestamp_ms=int(time.time()*1000)), timeout=5)
        print(f"[dryrun] gRPC 连接 OK")
    except Exception as e:
        print(f"[dryrun] gRPC 连接失败: {e}")
        return

    for round_i in range(args.rounds):
        print(f"\n{'='*50}\n=== Round {round_i+1}/{args.rounds} ===")

        # 1. 采集 + JPEG
        t_cam = time.time()
        images = []
        for name, cap in caps.items():
            for _ in range(2): cap.read()  # warmup
            ok, frame = cap.read()
            if not ok:
                print(f"  !! 相机 {name} 采集失败")
                continue
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            images.append(pb2.Image(camera_name=name, jpeg_data=buf.tobytes(), width=640, height=480))
        cam_ms = (time.time() - t_cam) * 1000
        print(f"  [采集] {cam_ms:.0f}ms, {len(images)} 路图像")

        # 2. gRPC 推理
        obs = pb2.Observation(
            timestamp_ms=int(time.time()*1000),
            instruction=INSTRUCTION,
            state=[0.0]*STATE_DIM,
            images=images,
        )
        t_inf = time.time()
        resp = stub.Predict(obs, timeout=30)
        inf_total_ms = (time.time() - t_inf) * 1000
        print(f"  [推理] 总 {inf_total_ms:.0f}ms (云端 {resp.inference_ms}ms), chunk={resp.chunk_size} dim={resp.action_dim}")

        # 3. 解析动作块
        actions = np.array(resp.actions).reshape(resp.chunk_size, resp.action_dim)
        print(f"  [动作] 块内 {len(actions)} 步，每步 {resp.action_dim} 维")

        # 4. 逐步送 motor_executor (DRY_RUN: 只记录+安全校验)
        for step in range(resp.chunk_size):
            step_actions = actions[step].tolist()
            executor.execute_action_chunk(step_actions, chunk_size=resp.chunk_size, action_dim=resp.action_dim)
            time.sleep(0.05)  # 模拟执行节奏

        # 5. 状态回读
        state = executor.get_state()
        smp = {k: round(v, 1) for k, v in list(state.items())[:5]}
        print(f"  [状态] 前5关节: {smp}")

    print(f"\n{'='*50}\nDRY_RUN 测试完成，共 {executor.action_count} 个动作被校验（未驱动电机）")

    # 释放
    for cap in caps.values():
        cap.release()


if __name__ == "__main__":
    main()