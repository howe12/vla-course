#!/usr/bin/env python3
"""云端 gRPC → NUC 电机 LIVE 执行器（真实驱动）

安全设计:
  1. 必须显式 --live 才能驱动电机
  2. 动作单位换算: 模型输出(0.1°) → lerobot 度数
  3. 全部安全机制串联: 限幅(-100~100) → 变化量截断 → 超时熔断
  4. 急停: 任何异常/KeyboardInterrupt 立即停止并断开

用法:
  python motor_executor_live.py --predict   # DRY_RUN 预测模式(默认)
  python motor_executor_live.py --live      # 真实驱动(需确认)
"""
import argparse
import sys
import time
import logging
import numpy as np
import cv2
import grpc

sys.path.insert(0, "/home/leo/vla-grpc/codes/step7_gemini/grpc")
import embodied_brain_pb2 as pb2
import embodied_brain_pb2_grpc as pb2_grpc
from motor_executor import MotorExecutor, SafetyConfig, JOINT_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live")

# ── 常量 ──
GRPC_ADDR = "127.0.0.1:50051"
CAMERA_MAP = [(0, "front"), (2, "left"), (5, "right")]
INSTRUCTION = "grab two objects into the middle box"
STATE_DIM = 14
ACTION_DIM = 14
CHUNK_SIZE = 5
# 模型输出单位: 0.1°；lerobot Sgr 期望: 度
MODEL_TO_DEG = 0.1
MODEL_TO_DEG_INV = 10.0  # 云端反归一化输出是"度"，executor 内部用 0.1° 单位
# 左/右臂串口（udev 软链接）
LEFT_PORT = "/dev/ttyACM_left_follower"
RIGHT_PORT = "/dev/ttyACM_right_follower"


class LiveMotorController:
    """真实电机控制器（基于 lerobot GeminiFollower）"""

    def __init__(self, calibrate: bool = True):
        from lerobot.robots.gemini_follower import GeminiFollower
        from lerobot.robots.gemini_follower.config_gemini_follower import GeminiFollowerConfig

        config = GeminiFollowerConfig(
            left_arm_port=LEFT_PORT,
            right_arm_port=RIGHT_PORT,
        )
        self.robot = GeminiFollower(config)
        # 校准 + 使能扭矩
        self.robot.connect(calibrate=calibrate)
        logger.info("✅ GeminiFollower 已连接并校准，扭矩已使能")
        # 等待电机就绪（首次读取前需要 settle）
        time.sleep(1.5)

    def read_state_0p1deg(self) -> list[float]:
        """读取当前 14 关节位置，转为 0.1° 单位（对齐训练数据语义）

        lerobot 返回度数 → ×10 → 0.1° 单位
        """
        obs = self.robot.get_observation()
        pos = {}
        for k, v in obs.items():
            if k.endswith(".pos"):
                name = k.removesuffix(".pos")
                pos[name] = round(v * 10, 1)  # 度 → 0.1°
        # 按 JOINT_NAMES 顺序输出
        return [pos.get(name, 0.0) for name in JOINT_NAMES]

    def send_actions(self, targets_0p1deg: dict[str, float]):
        """发送14维目标位置（0.1°单位）到双臂"""
        # 转为 lerobot 期望格式 {motor.pos: 度数}
        action = {}
        for name, val_0p1 in targets_0p1deg.items():
            action[f"{name}.pos"] = val_0p1 * MODEL_TO_DEG
        self.robot.send_action(action)

    def emergency_stop(self):
        logger.warning("🛑 急停：关闭扭矩并断开")
        try:
            self.robot.disconnect()
        except Exception as e:
            logger.error(f"断开失败: {e}")
        logger.warning("🛑 已断开，电机扭矩已关闭")

    def close(self):
        try:
            self.robot.disconnect()
            logger.info("已安全断开")
        except Exception as e:
            logger.error(f"断开失败: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="真实驱动电机（默认 DRY_RUN）")
    ap.add_argument("--rounds", type=int, default=10, help="控制回合数")
    ap.add_argument("--max-joint-delta", type=float, default=20.0, help="单步最大变化量(0.1°单位)")
    ap.add_argument("--calibrate", type=bool, default=True, help="连接时是否校准")
    ap.add_argument("--step-delay", type=float, default=0.3, help="每步执行间隔(秒)")
    args = ap.parse_args()

    # 安全确认
    if args.live:
        print("╔══════════════════════════════════════════════════════╗")
        print("║  ⚠️ LIVE 模式：将真实驱动 LEO-Gemini 双臂电机！      ║")
        print("║  请确认：                                           ║")
        print("║  1. 机器周围无障碍物、无人员                            ║")
        print("║  2. 急停按钮可达                                      ║")
        print("║  3. 夹爪/机械结构处于安全位置                          ║")
        print("╚══════════════════════════════════════════════════════╝")
        resp = input("输入 'ENABLE' 确认驱动电机: ").strip()
        if resp != "ENABLE":
            print("已取消，未驱动电机。")
            return

    # 连接 gRPC
    channel = grpc.insecure_channel(GRPC_ADDR)
    stub = pb2_grpc.BrainServiceStub(channel)

    # 安全执行器（限幅 + 变化量截断）
    executor = MotorExecutor(SafetyConfig(
        dry_run=not args.live,
        max_joint_delta=args.max_joint_delta,
    ))

    # LIVE 控制器
    live = None
    if args.live:
        live = LiveMotorController(calibrate=args.calibrate)
        # 注入 LIVE 驱动回调：安全校验通过后由 executor 调用
        executor.live_driver = live.send_actions
        # 初始位置对齐：读取当前姿态作为推理起点
        init_state = live.read_state_0p1deg()
        logger.info(f"🎯 初始位置对齐 (0.1°): {[round(v) for v in init_state]}")
        # 让安全执行器从当前实际位置开始（而非 0），避免首步大幅跳变
        executor.last_positions = dict(zip(JOINT_NAMES, init_state))
    else:
        init_state = [0.0] * STATE_DIM

    # 开相机
    caps = {}
    for idx, name in CAMERA_MAP:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        caps[name] = cap

    try:
        # Ping
        stub.Ping(pb2.PingRequest(timestamp_ms=int(time.time()*1000)), timeout=5)
        logger.info("gRPC 连接 OK")

        for round_i in range(args.rounds):
            # 1. 采集
            images = []
            for name, cap in caps.items():
                for _ in range(2): cap.read()
                ok, frame = cap.read()
                if not ok:
                    logger.warning(f"相机 {name} 采集失败")
                    continue
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                images.append(pb2.Image(camera_name=name, jpeg_data=buf.tobytes(), width=640, height=480))

            # 2. 推理（state 用当前实际姿态，实现闭环）
            current_state = live.read_state_0p1deg() if live is not None else init_state
            obs = pb2.Observation(
                timestamp_ms=int(time.time()*1000),
                instruction=INSTRUCTION,
                state=current_state,
                images=images,
            )
            resp = stub.Predict(obs, timeout=30)
            actions = np.array(resp.actions).reshape(resp.chunk_size, resp.action_dim)
            logger.info(f"Round {round_i+1}: 云端推理 {resp.inference_ms}ms, 动作块 {resp.chunk_size}x{resp.action_dim}")

            # 3. 逐步执行（安全校验在 executor 内，LIVE 时自动调用 live_driver）
            # 云端反归一化输出单位为"度"，executor 内部用 0.1° 单位，需 ×10
            for step in range(resp.chunk_size):
                step_actions = [v * MODEL_TO_DEG_INV for v in actions[step].tolist()]
                executor.execute_action_chunk(step_actions, chunk_size=resp.chunk_size, action_dim=resp.action_dim)
                time.sleep(args.step_delay)

        logger.info("控制循环完成")
    except grpc.RpcError as e:
        logger.error(f"gRPC 错误: {e}")
        if live: live.emergency_stop()
    except KeyboardInterrupt:
        logger.warning("中断！")
        if live: live.emergency_stop()
    except Exception as e:
        logger.error(f"异常: {e}", exc_info=True)
        if live: live.emergency_stop()
    finally:
        for cap in caps.values():
            cap.release()
        if live:
            live.close()


if __name__ == "__main__":
    main()