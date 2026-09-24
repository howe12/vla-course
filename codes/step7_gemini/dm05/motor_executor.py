#!/usr/bin/env python3
"""Orin 电机执行模块（DRY_RUN 安全模式）

接收 gRPC 返回的 action chunk，解析为关节目标位置，
应用安全限幅后，DRY_RUN 模式下只记录到日志，不发送串口指令。

电机结构（gemini_follower = 2 × SgrFollower）：
  左臂: left_waist, left_shoulder, left_elbow, left_forearm_roll,
        left_wrist_flex, left_wrist_roll, left_gripper
  右臂: right_waist, right_shoulder, right_elbow, right_forearm_roll,
        right_wrist_flex, right_wrist_roll, right_gripper

归一化范围: RANGE_M100_100 (-100 ~ 100)
位置单位: 0.1 度 (int16 / 10)
"""
import time
import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("executor")

# 关节名称（对齐 gemini_follower 数据集 action 前 14 维）
JOINT_NAMES = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
]

# 安全限位 —— 双层设计
#
# ⚠️ 单位: 0.1 度 (int16 / 10) —— 必须与 execute_action_chunk 的输入单位一致！
#
# 历史 bug（已修复）: 原值为 (-100, 100)，但那是"度"的量纲，而
# execute_action_chunk 收到的是 0.1° 单位（如 shoulder=835 表示 83.5°）。
# 结果所有关节目标被 clamp 到 ±100 (= ±10°)，机械臂被拉向 ±10° 附近，
# 并产生"变化量 740 超限"这类假告警（因为 target 被改成了 10°）。
#
# ── SAFE_LIMITS: 训练数据实测范围 + 约 4° 余量（默认，安全优先）──
#   实测来源: 25851 帧 LEO-Gemini 数据集逐关节 min/max
#   L_shldr[-64.1,95.1] R_shldr[-62.6,104.5]  L_elbow[-90.0,65.9] R_elbow[-88.3,71.3]
#   L_wflex[-96.8,18.0] R_wflex[-88.5,3.8]    L_wroll[-5.4,51.7]  R_wroll[-44.2,61.5]
#   L_grip[-80.0,64.9]  R_grip[-79.9,66.3]    L_waist[-5.9,47.1]  R_waist[-55.5,1.8]
#   L_farm[-37.7,10.8]  R_farm[-17.7,16.1]
#   注意: 夹爪必须用 min/max 而非分位数 —— 夹爪 <0（闭合）仅占 0.13%，
#         分位数会把它排除，导致夹爪无法闭合抓取。
SAFE_LIMITS = {
    "waist": (-600, 510),
    "shoulder": (-670, 1085),
    "elbow": (-940, 755),
    "forearm_roll": (-420, 200),
    "wrist_flex": (-1010, 220),
    "wrist_roll": (-480, 660),
    "gripper": (-840, 705),
}

# ── LOOSE_LIMITS: 更宽的机械安全边界（--loose-limits 启用）──
#   仅在确认环境安全、需要更大活动空间时使用
LOOSE_LIMITS = {
    "waist": (-700, 600),
    "shoulder": (-800, 1200),
    "elbow": (-1000, 800),
    "forearm_roll": (-500, 300),
    "wrist_flex": (-1100, 300),
    "wrist_roll": (-600, 700),
    "gripper": (-900, 800),
}

# 默认使用 SAFE_LIMITS
DEFAULT_LIMITS = SAFE_LIMITS


@dataclass
class SafetyConfig:
    """安全配置"""
    dry_run: bool = True                    # True=只记录不驱动, False=真实驱动(需授权)
    max_joint_delta: float = 20.0           # 单步最大关节变化量（归一化单位）
    action_timeout_ms: int = 10000          # 动作超时（DM0.5 chunk=50 需较大值）
    enable_clipping: bool = True            # 是否启用限幅
    log_actions: bool = True                # 是否记录动作到日志
    limits: dict | None = None              # 关节限位表(0.1度单位), None=用 DEFAULT_LIMITS


class MotorExecutor:
    """电机执行器（DRY_RUN 安全模式）"""

    def __init__(self, safety_config: SafetyConfig = None):
        self.config = safety_config or SafetyConfig()
        self.last_positions = {name: 0.0 for name in JOINT_NAMES}
        self.last_action_time = time.time()
        self.action_count = 0
        self._robot = None  # LIVE 模式下的 lerobot Robot 实例

        mode = "DRY_RUN（只记录）" if self.config.dry_run else "⚠️ LIVE（真实驱动）"
        logger.info(f"MotorExecutor 初始化，模式: {mode}")
        if not self.config.dry_run:
            logger.warning("⚠️ LIVE 模式：将真实驱动电机！请确认安全！")

    def execute_action_chunk(self, actions: list[float], chunk_size: int, action_dim: int):
        """执行一个 action chunk

        Args:
            actions: 展平的动作序列 [chunk_size * action_dim]
            chunk_size: 动作步数
            action_dim: 每步动作维度（gemini=14 或 18）
        """
        now = time.time()
        # 超时检测（首次执行跳过，避免初始化延迟误判）
        if self.action_count > 0:
            elapsed_ms = (now - self.last_action_time) * 1000
            if elapsed_ms > self.config.action_timeout_ms:
                logger.warning(f"动作超时: {elapsed_ms:.0f}ms > {self.config.action_timeout_ms}ms，跳过")
                return

        self.last_action_time = now
        self.action_count += 1

        # 只取前 14 维（双臂关节），忽略底盘速度维（如果有）
        joint_dim = min(action_dim, len(JOINT_NAMES))

        # 解析第一步动作（n_action_steps 控制实际执行几步）
        step_actions = actions[:joint_dim]
        targets = self._parse_targets(step_actions, joint_dim)

        # 安全校验
        safe_targets = self._apply_safety(targets)
        if safe_targets is None:
            return  # 安全校验失败，拒绝执行

        # 执行（DRY_RUN 只记录）
        if self.config.dry_run:
            self._log_dry_run(safe_targets)
        else:
            self._execute_live(safe_targets)

        # 更新状态
        self.last_positions = safe_targets

    def _parse_targets(self, step_actions: list[float], joint_dim: int) -> dict[str, float]:
        """解析动作向量为关节目标位置"""
        targets = {}
        for i, name in enumerate(JOINT_NAMES[:joint_dim]):
            targets[name] = float(step_actions[i]) if i < len(step_actions) else self.last_positions[name]
        return targets

    def _apply_safety(self, targets: dict[str, float]) -> dict[str, float] | None:
        """应用安全限幅和变化量限制"""
        safe = {}
        for name, target in targets.items():
            # 1. 关节限位
            if self.config.enable_clipping:
                limits_table = self.config.limits or DEFAULT_LIMITS
                for jt in limits_table:
                    if jt in name:
                        lo, hi = limits_table[jt]
                        clamped = max(lo, min(hi, target))
                        if clamped != target:
                            logger.warning(
                                f"关节 {name} 目标 {target/10:.1f}° 越界 "
                                f"[{lo/10:.1f}°, {hi/10:.1f}°]，截断到 {clamped/10:.1f}°"
                            )
                        target = clamped
                        break

            # 2. 单步变化量限制
            delta = abs(target - self.last_positions.get(name, 0.0))
            if delta > self.config.max_joint_delta:
                logger.warning(
                    f"关节 {name} 变化量 {delta:.1f} 超限（max {self.config.max_joint_delta}），"
                    f"截断到 {self.config.max_joint_delta}"
                )
                direction = 1.0 if target > self.last_positions[name] else -1.0
                target = self.last_positions[name] + direction * self.config.max_joint_delta

            safe[name] = target
        return safe

    def _log_dry_run(self, targets: dict[str, float]):
        """DRY_RUN 模式：记录动作到日志"""
        if not self.config.log_actions:
            return
        parts = [f"{name}={val:.2f}" for name, val in targets.items()]
        logger.info(f"[DRY_RUN #{self.action_count}] 目标位置: {', '.join(parts)}")

    def _execute_live(self, targets: dict[str, float]):
        """LIVE 模式：通过注入的 live_driver 回调驱动电机"""
        if not hasattr(self, 'live_driver') or self.live_driver is None:
            raise RuntimeError("LIVE 驱动回调未注入 (executor.live_driver = ...)")
        self.live_driver(targets)

    def get_state(self) -> dict[str, float]:
        """获取当前关节状态（DRY_RUN 返回上次目标，LIVE 读取真实位置）"""
        if self.config.dry_run:
            return dict(self.last_positions)
        # TODO: LIVE 模式读取 bus.sync_read("Present_Position")
        return dict(self.last_positions)

    def emergency_stop(self):
        """急停：禁用扭矩（LIVE 模式）"""
        logger.warning("🛑 急停触发！")
        if not self.config.dry_run and self._robot:
            # TODO: 禁用扭矩
            pass
        self.last_positions = {name: 0.0 for name in JOINT_NAMES}


if __name__ == "__main__":
    # 独立测试：模拟接收 action chunk
    executor = MotorExecutor(SafetyConfig(dry_run=True, max_joint_delta=15.0))

    # 模拟云端返回的 action chunk（14 维关节 × 50 步）
    import random
    for step in range(5):
        fake_actions = [random.uniform(-30, 30) for _ in range(14)]
        executor.execute_action_chunk(fake_actions, chunk_size=50, action_dim=14)
        time.sleep(0.1)

    print(f"\n最终状态: {executor.get_state()}")
    print("DRY_RUN 测试完成，未驱动任何电机。")
