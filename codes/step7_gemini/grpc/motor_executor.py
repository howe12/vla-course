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

# 安全限位（归一化值域 -100 ~ 100，对应 RANGE_M100_100）
# 实际部署时应从 lerobot 校准文件读取每个关节的真实范围
DEFAULT_LIMITS = {
    "waist": (-100, 100),
    "shoulder": (-100, 100),
    "elbow": (-100, 100),
    "forearm_roll": (-100, 100),
    "wrist_flex": (-100, 100),
    "wrist_roll": (-100, 100),
    "gripper": (-100, 100),  # 夹爪
}


@dataclass
class SafetyConfig:
    """安全配置"""
    dry_run: bool = True                    # True=只记录不驱动, False=真实驱动(需授权)
    max_joint_delta: float = 20.0           # 单步最大关节变化量（归一化单位）
    action_timeout_ms: int = 2000           # 动作超时（超过则停止）
    enable_clipping: bool = True            # 是否启用限幅
    log_actions: bool = True                # 是否记录动作到日志


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
        # 超时检测
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
                joint_type = name.split("_", 1)[1] if "_" in name else name
                # 去掉 left_/right_ 前缀获取关节类型
                for jt in DEFAULT_LIMITS:
                    if jt in name:
                        lo, hi = DEFAULT_LIMITS[jt]
                        target = max(lo, min(hi, target))
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
        """LIVE 模式：真实驱动电机（需授权，当前未实现）"""
        # TODO: 用户授权后实现
        # 1. 初始化 lerobot GeminiFollower
        # 2. 归一化值 → 电机原始位置
        # 3. bus.sync_write("Goal_Position", goal_pos)
        raise NotImplementedError(
            "LIVE 模式未实现。需要用户明确授权后才能启用真实电机驱动。"
        )

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
