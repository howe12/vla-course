#!/usr/bin/env python3
"""DM0.5 HTTP → NUC 电机 LIVE/DRY_RUN 执行器

基于 SmolVLA motor_executor_live.py 适配:
  - gRPC → HTTP (brain_http_client)
  - state 单位: 度 (不 ×10)
  - chunk_size: 50 (非 5)
  - step_delay: 0.05s (50步 × 0.05 = 2.5s/chunk)

安全设计 (继承自 SmolVLA):
  1. 必须显式 --live 才能驱动电机
  2. MotorExecutor 三层防护: 限幅 → 变化量截断 → 超时熔断
  3. 急停: 任何异常/KeyboardInterrupt 立即停止并断开
  4. 初始位置对齐: 从当前真实姿态开始

用法:
  python motor_executor_live_dm05.py                     # DRY_RUN (默认)
  python motor_executor_live_dm05.py --live              # 真实驱动
  python motor_executor_live_dm05.py --rounds 5          # 指定回合数
"""
import argparse
import sys
import os
import time
import logging
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain_http_client import DM05HTTPClient, CHUNK_SIZE, ACTION_DIM
from motor_executor import MotorExecutor, SafetyConfig, JOINT_NAMES, SAFE_LIMITS, LOOSE_LIMITS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dm05_live")

# ── 常量 ──
HTTP_ADDR = "http://127.0.0.1:7891"
CAMERA_MAP = [(0, "front"), (4, "left"), (2, "right")]
INSTRUCTION = "grab two objects into the middle box"
STATE_DIM = 14
# DM0.5 输出单位: 度; executor 内部用 0.1° 单位
DEG_TO_0P1DEG = 10.0
# 左/右臂串口（udev 软链接）
LEFT_PORT = "/dev/ttyACM_left_follower"
RIGHT_PORT = "/dev/ttyACM_right_follower"


# 训练数据的初始位姿（LEO-Gemini 数据集 30 集首帧的中位数, 单位: 度）
#
# 为什么需要归位到初始位姿：
#   模型是在「特定起始姿态」下采集的示教数据上训练的。若部署时机器人的静止
#   姿态与训练起始姿态不一致，模型就会在分布外状态下推理，输出不可靠。
#   实测差异最大处：right_shoulder 104.3→89.0(训练中 [85,90) 占 61.44%，
#   而 104.3 仅占 0.07%)，导致右臂被错误驱动（摆幅 124.6°）。
#   实测归位右肩后：右臂摆幅 -31%，左臂朝抓取位推进 +32~35°。
TRAINING_INITIAL_POSE = {
    "left_waist": 0.8,
    "left_shoulder": 84.5,
    "left_elbow": -89.7,
    "left_forearm_roll": -8.3,
    "left_wrist_flex": -13.7,
    "left_wrist_roll": 9.5,
    "left_gripper": 15.2,
    "right_waist": -2.1,
    "right_shoulder": 89.0,
    "right_elbow": -87.6,
    "right_forearm_roll": -5.2,
    "right_wrist_flex": -17.2,
    "right_wrist_roll": 5.5,
    "right_gripper": 15.7,
}


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
        self.robot.connect(calibrate=calibrate)
        logger.info("✅ GeminiFollower 已连接并校准，扭矩已使能")
        time.sleep(1.5)  # 等待电机就绪

    def read_state_deg(self) -> list[float]:
        """读取当前 14 关节位置，单位: 度 (直接用于 DM0.5 输入)

        lerobot 返回度数，直接使用。
        """
        obs = self.robot.get_observation()
        pos = {}
        for k, v in obs.items():
            if k.endswith(".pos"):
                name = k.removesuffix(".pos")
                pos[name] = float(v)  # 度，不转换
        return [pos.get(name, 0.0) for name in JOINT_NAMES]

    def send_actions(self, targets_0p1deg: dict[str, float]):
        """发送14维目标位置（0.1°单位）到双臂"""
        action = {}
        for name, val_0p1 in targets_0p1deg.items():
            action[f"{name}.pos"] = val_0p1 * 0.1  # 0.1° → 度
        self.robot.send_action(action)

    def open_grippers(self, target_deg: float = 20.0, step_deg: float = 1.5,
                      tol_deg: float = 0.8, timeout_s: float = 15.0):
        """把双夹爪缓慢开到 target_deg，其余关节保持不动。

        为什么必须做：
          训练数据中夹爪落在 <0 区间的帧仅 34/25851 = 0.13%，属极端边缘分布。
          机器人静止/重连后夹爪因弹簧停在 -80（闭合），模型在该点会输出约 +6~+12
          （回归分布主体），而非真值 -80，造成 86~92° 的绝对位置偏差。
          实测：夹爪 -80 起步时 chunk[0] 误差 13.24°/92.2°，
                夹爪 +20 起步时降为 0.67°/3.6°（20~26 倍改善）。

        注意：disconnect() 会关闭扭矩使夹爪弹回闭合，故每次重连后都需重新执行。
        """
        t0 = time.time()
        cur = self.read_state_deg()
        logger.info(f"🔧 夹爪归位: L={cur[6]:.1f}° R={cur[13]:.1f}° → 目标 {target_deg:.1f}°")
        while time.time() - t0 < timeout_s:
            action = {f"{name}.pos": float(cur[i]) for i, name in enumerate(JOINT_NAMES)}
            done = True
            for idx in (6, 13):  # left_gripper, right_gripper
                delta = target_deg - cur[idx]
                if abs(delta) > tol_deg:
                    action[f"{JOINT_NAMES[idx]}.pos"] = float(
                        cur[idx] + (1.0 if delta > 0 else -1.0) * min(abs(delta), step_deg)
                    )
                    done = False
            if done:
                break
            self.robot.send_action(action)
            time.sleep(0.05)
            cur = self.read_state_deg()
        logger.info(f"✅ 夹爪就位: L={cur[6]:.1f}° R={cur[13]:.1f}° (耗时 {time.time()-t0:.1f}s)")
        return cur

    def home_joints(self, targets: dict[str, float], step_deg: float = 5.0,
                    tol_deg: float = 1.5, timeout_s: float = 180.0,
                    settle_s: float = 1.2):
        """把指定关节移动到目标角度，其余关节保持不动。

        ⚠️ 实现要点：read_state_deg() 读的是后台缓存，实测更新率仅约 3 Hz。
        若用「每 50ms 发一次 cur+1.5°」的渐进方式，在缓存未更新期间会反复重发
        同一目标，实际推进极慢（实测 40s 仅走 5.8°）。
        故改为【分段直发 + 等待到位】：每段最多走 step_deg，发一次后等 settle_s
        让电机真正到位（实测 10° 约 0.6s），再进入下一段。

        ⚠️ 同步而非串行：早期版本对每个关节各自 while 循环（串行），
        结果每个关节都要独立等待，14 个关节总耗时爆炸；更糟的是只要有一个
        关节卡在 tol 边缘，其余关节就永远轮不到，实测 90s 只让 4 个关节动了
        不到 3°。现改为【所有待归位关节同步推进】：每段让所有 todo 关节各走
        step_deg，段数 = max(偏差)/step_deg，与关节数无关。

        为什么需要归位（以 right_shoulder 为例）：
          训练数据各集首帧 R_shldr 均值 89.4°，直方图显示 [85,90) 占 61.44%，
          而 [90,110) 仅 17 帧 / 0.07%。机器人断电静止位却是 104.7°，落在
          0.07% 边缘区，导致模型在 OOD 起始状态下错误驱动右臂（摆幅 124.6°），
          并形成训练数据中不存在的「左臂抓取位+右臂移动中」组合。
          实测归位后右臂摆幅降至 51.6°（-59%）。

        Args:
            targets: {关节名(JOINT_NAMES 中的名字): 目标角度(度)}
        """
        t0 = time.time()
        cur = self.read_state_deg()
        idxs = {JOINT_NAMES.index(n): t for n, t in targets.items() if n in JOINT_NAMES}
        if not idxs:
            return cur
        desc = ", ".join(f"{JOINT_NAMES[i]}={cur[i]:.1f}→{t:.1f}°" for i, t in idxs.items())
        logger.info(f"🔧 关节归位: {desc}")

        seg = 0
        while time.time() - t0 < timeout_s:
            cur = self.read_state_deg()
            todo = {i: t for i, t in idxs.items() if abs(t - cur[i]) > tol_deg}
            if not todo:
                break
            # 关键：action 必须包含全部 14 个关节。未列入 todo 的关节填当前位置，
            # 让它们保持不动（不填则该关节无目标，会被驱动层当作 0 处理）。
            action = {f"{n}.pos": float(cur[i]) for i, n in enumerate(JOINT_NAMES)}
            for i, t in todo.items():
                d = t - cur[i]
                action[f"{JOINT_NAMES[i]}.pos"] = float(
                    cur[i] + (1.0 if d > 0 else -1.0) * min(abs(d), step_deg)
                )
            self.robot.send_action(action)
            seg += 1
            time.sleep(settle_s)   # 等本段移动完成（缓存 ~3Hz，1s 足够覆盖）

        cur = self.read_state_deg()
        stuck = {JOINT_NAMES[i]: round(t - cur[i], 1)
                 for i, t in idxs.items() if abs(t - cur[i]) > tol_deg * 2}
        if stuck:
            logger.warning(f"⚠️ 归位后仍有偏差: {stuck}")
        now = ", ".join(f"{JOINT_NAMES[i]}={cur[i]:.1f}°" for i in idxs)
        logger.info(f"✅ 关节就位: {now} ({seg} 段, 耗时 {time.time()-t0:.1f}s)")
        return cur

    def home_all(self, step_deg: float = 5.0, tol_deg: float = 1.5,
                 timeout_s: float = 180.0, settle_s: float = 1.2):
        """整机（14 关节）同步归位到 TRAINING_INITIAL_POSE。

        收敛判据与推进逻辑复用 home_joints（同步分段推进），此处仅补齐目标集。
        """
        targets = {n: TRAINING_INITIAL_POSE[n] for n in JOINT_NAMES}
        cur0 = self.read_state_deg()
        err0 = max(abs(targets[n] - cur0[i]) for i, n in enumerate(JOINT_NAMES))
        logger.info(f"🔧 整机归位: 最大偏差 {err0:.1f}° → 训练初始位姿(14关节)")
        cur = self.home_joints(targets, step_deg=step_deg, tol_deg=tol_deg,
                               timeout_s=timeout_s, settle_s=settle_s)
        err1 = max(abs(targets[n] - cur[i]) for i, n in enumerate(JOINT_NAMES))
        logger.info(f"✅ 整机就位: 最大偏差 {err0:.1f}° → {err1:.1f}°")
        return cur

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
    ap = argparse.ArgumentParser(description="DM0.5 LIVE/DRY_RUN 电机执行器")
    ap.add_argument("--live", action="store_true", help="真实驱动电机（默认 DRY_RUN）")
    ap.add_argument("--rounds", type=int, default=5, help="控制回合数")
    ap.add_argument("--max-joint-delta", type=float, default=20.0,
                    help="单步最大变化量 (0.1°单位, 20=2°)")
    ap.add_argument("--calibrate", type=bool, default=True, help="连接时是否校准")
    ap.add_argument("--step-delay", type=float, default=0.05,
                    help="每步执行间隔 (秒), DM0.5 chunk=50 建议 0.05")
    ap.add_argument("--addr", default=HTTP_ADDR, help="DM0.5 推理服务地址")
    ap.add_argument("--home-mode", choices=["initial", "grip", "off"], default="initial",
                    help="启动归位模式: initial=归位到训练初始位姿(推荐, 全14关节); "
                         "grip=仅归位夹爪; off=不归位")
    ap.add_argument("--grip-open", type=float, default=20.0,
                    help="启动时把夹爪开到的角度(度)。训练分布主体在 [0,+65]，"
                         "夹爪闭合(-80)属 0.13%% 极端边缘，会导致模型输出错误。"
                         "设 -999 跳过(不推荐)")
    ap.add_argument("--home-right-shoulder", type=float, default=89.0,
                    help="启动时把 right_shoulder 归位到的角度(度)。训练首帧均值 89.4，"
                         "[85,90) 占 61%%。机器人断电静止位为 104.7(训练中仅 0.07%%)，"
                         "属分布外，会导致右臂被错误驱动。设 -999 跳过")
    ap.add_argument("--flush-camera", action="store_true",
                    help="开相机后先排空缓冲区，取真正最新帧（默认关闭以保持原有行为）")
    ap.add_argument("--loose-limits", action="store_true",
                    help="使用更宽的机械安全限位（默认用训练数据范围 SAFE_LIMITS）")
    ap.add_argument("--max-speed-warn", type=float, default=25.0,
                    help="峰值速度告警阈值(度/秒)。理论上限 = max_joint_delta/10/step_delay")
    ap.add_argument("--max-delta0-warn", type=float, default=15.0,
                    help="异常轮次保护阈值：chunk[0] 与当前 state 的最大偏差超过该值"
                         "(度)时判定观测异常，跳过本轮执行")
    args = ap.parse_args()

    # ── 安全确认 ──
    if args.live:
        print("╔══════════════════════════════════════════════════════╗")
        print("║  ⚠️  LIVE 模式：将真实驱动 LEO-Gemini 双臂电机！     ║")
        print("║  模型: DM0.5 LoRA checkpoint-500                     ║")
        print("║  Chunk: 50 steps × step-delay                        ║")
        print("║  请确认：                                             ║")
        print("║  1. 机器周围无障碍物、无人员                          ║")
        print("║  2. 急停按钮可达                                      ║")
        if args.home_mode == "initial":
            print("║  3. 整机将归位到训练初始位姿(14关节)               ║")
            print("║     L_wflex -36→-14, R_shldr 104→89, 夹爪 -80→+16 ║")
            print("║     请确认机械臂活动范围内无障碍物                  ║")
        else:
            print(f"║  3. 夹爪张开路径({args.grip_open:+.0f}°)及周边无障碍物      ║")
        print("╚══════════════════════════════════════════════════════╝")
        resp = input("输入 'ENABLE' 确认驱动电机: ").strip()
        if resp != "ENABLE":
            print("已取消，未驱动电机。")
            return

    # ── 连接推理服务 ──
    client = DM05HTTPClient(args.addr)
    if not client.ping():
        logger.error(f"无法连接 DM0.5 推理服务: {args.addr}")
        logger.info("提示: 确保 SSH 隧道已建立 (tunnel_dm05.sh)")
        return
    logger.info(f"DM0.5 推理服务连接 OK: {args.addr}")

    # ── 安全执行器 ──
    limits = LOOSE_LIMITS if args.loose_limits else SAFE_LIMITS
    logger.info(
        "关节限位模式: " + ("LOOSE(机械边界)" if args.loose_limits else "SAFE(训练范围)")
    )
    executor = MotorExecutor(SafetyConfig(
        dry_run=not args.live,
        max_joint_delta=args.max_joint_delta,
        limits=limits,
    ))

    # ── LIVE 控制器 ──
    live = None
    if args.live:
        live = LiveMotorController(calibrate=args.calibrate)
        executor.live_driver = live.send_actions
        # 启动归位：必须早于初始位置对齐（对齐需用到归位后的姿态）
        if args.home_mode == "initial":
            logger.info("归位模式: initial (训练数据初始位姿, 14 关节同步)")
            live.home_all(step_deg=5.0, tol_deg=1.5, timeout_s=180.0, settle_s=1.2)
        elif args.home_mode == "grip":
            logger.info("归位模式: grip (仅夹爪)")
            if args.grip_open > -900:
                live.open_grippers(target_deg=args.grip_open)
            if args.home_right_shoulder > -900:
                live.home_joints({"right_shoulder": args.home_right_shoulder},
                                 step_deg=5.0, timeout_s=60.0)
        else:
            logger.warning("归位模式: off (不做任何归位)")
        # 初始位置对齐
        init_state_deg = live.read_state_deg()
        init_state_0p1 = [v * DEG_TO_0P1DEG for v in init_state_deg]
        logger.info(f"🎯 初始位置对齐 (度): {[round(v, 1) for v in init_state_deg]}")
        executor.last_positions = dict(zip(JOINT_NAMES, init_state_0p1))
    else:
        init_state_deg = [0.0] * STATE_DIM

    # ── 开相机 ──
    caps = {}
    for idx, name in CAMERA_MAP:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if args.flush_camera:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            caps[name] = cap
            logger.info(f"相机 {name} (video{idx}) 已打开")
        else:
            logger.warning(f"相机 {name} (video{idx}) 打开失败")

    if len(caps) < 3:
        logger.warning(f"仅 {len(caps)}/3 路相机可用，缺失的将使用空图")

    # 预热：夹爪归位(~10s)/开相机期间驱动缓冲里可能是旧帧，先丢弃若干帧
    # （实测不预热时首轮 delta0 max 可达 14.4°，属瞬态）
    for _name, _cap in caps.items():
        for _ in range(6):
            _cap.read()
    logger.info(f"相机预热完成，缓冲已刷新")

    try:
        for round_i in range(args.rounds):
            logger.info(f"\n{'='*50}\n=== Round {round_i+1}/{args.rounds} ===")

            # 1. 采集图像
            t_cam = time.time()
            images_jpeg = {}
            for name, cap in caps.items():
                if args.flush_camera:
                    # 排空驱动缓冲，确保取到真正最新帧
                    for _ in range(4):
                        cap.grab()
                    ok, frame = cap.retrieve()
                else:
                    for _ in range(2):
                        cap.read()  # warmup
                    ok, frame = cap.read()
                if not ok:
                    logger.warning(f"相机 {name} 采集失败")
                    continue
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                images_jpeg[name] = buf.tobytes()
            cam_ms = (time.time() - t_cam) * 1000
            logger.info(f"[采集] {cam_ms:.0f}ms, {len(images_jpeg)} 路图像")

            # 2. 读取当前 state (闭环)
            if live is not None:
                current_state_deg = live.read_state_deg()
            else:
                current_state_deg = init_state_deg

            # 3. HTTP 推理
            t_inf = time.time()
            actions, server_latency = client.predict(
                state_deg=current_state_deg,
                images_jpeg=images_jpeg,
                instruction=INSTRUCTION,
            )
            total_ms = (time.time() - t_inf) * 1000
            logger.info(
                f"[推理] 总 {total_ms:.0f}ms (云端 {server_latency:.0f}ms), "
                f"chunk={actions.shape}"
            )

            # 诊断：打印 state / chunk[0] / 逐维偏差，便于定位异常
            d0 = actions[0] - np.asarray(current_state_deg, dtype=np.float32)
            worst = int(np.argmax(np.abs(d0)))
            logger.info(
                f"[诊断] state={[round(v,1) for v in current_state_deg]}"
            )
            logger.info(
                f"[诊断] chunk0={[round(float(v),1) for v in actions[0]]}"
            )
            logger.info(
                f"[诊断] delta0 均={np.abs(d0).mean():.2f}° "
                f"max={np.abs(d0).max():.1f}°({JOINT_NAMES[worst]})"
            )

            # 异常轮次保护：chunk[0] 与当前观测偏离过大说明本帧观测/状态不可靠
            # （实测偶发过一次 delta0 达 76.7°，成因未完全定位；跳过该轮更安全）
            if np.abs(d0).max() > args.max_delta0_warn:
                logger.warning(
                    f"⚠️ 本轮 chunk0 偏离过大 (max {np.abs(d0).max():.1f}° > "
                    f"{args.max_delta0_warn:.0f}°)，判定为异常观测，跳过本轮执行"
                )
                continue

            # 4. 逐步执行 (度 → 0.1° → MotorExecutor 安全校验)
            t_exec_start = time.time()
            for step in range(actions.shape[0]):
                step_actions_0p1 = [v * DEG_TO_0P1DEG for v in actions[step].tolist()]
                executor.execute_action_chunk(
                    step_actions_0p1,
                    chunk_size=CHUNK_SIZE,
                    action_dim=ACTION_DIM,
                )
                time.sleep(args.step_delay)

            # 执行后统计实际运动速度与行程（安全监控）
            if live is not None:
                after_deg = live.read_state_deg()
                travel = np.abs(
                    np.asarray(after_deg, dtype=np.float32)
                    - np.asarray(current_state_deg, dtype=np.float32)
                )
                exec_sec = max(time.time() - t_exec_start, 1e-6)
                peak = float(travel.max())
                peak_joint = JOINT_NAMES[int(np.argmax(travel))]
                speed = peak / exec_sec
                logger.info(
                    f"[安全] 本轮行程 max={peak:.1f}°({peak_joint}) "
                    f"均={travel.mean():.1f}° | 峰值速度≈{speed:.1f}°/s "
                    f"| 耗时 {exec_sec:.2f}s"
                )
                if speed > args.max_speed_warn:
                    logger.warning(
                        f"⚠️ 峰值速度 {speed:.1f}°/s 超过告警阈值 "
                        f"{args.max_speed_warn:.0f}°/s，建议降低 --max-joint-delta"
                    )

            logger.info(f"[完成] Round {round_i+1}, 累计 {executor.action_count} 步")

        logger.info(f"\n控制循环完成，共 {executor.action_count} 步")

    except ConnectionError as e:
        logger.error(f"连接错误: {e}")
        if live:
            live.emergency_stop()
    except KeyboardInterrupt:
        logger.warning("中断！")
        if live:
            live.emergency_stop()
    except Exception as e:
        logger.error(f"异常: {e}", exc_info=True)
        if live:
            live.emergency_stop()
    finally:
        for cap in caps.values():
            cap.release()
        if live:
            live.close()


if __name__ == "__main__":
    main()
