#!/usr/bin/env python3
"""ch1 入门实验 1/2：动作离散化 — 理解 256 桶方案

VLA 把连续动作映射到 256 个离散桶，让 LLM 能像预测"下一个词"一样预测"下一个动作"。

这个实验不加载任何模型——纯数学演示。

运行：
    cd /develop/vla-course/codes && source .venv/bin/activate
    python step1_basics/action_discretize.py
"""

import numpy as np

print("=" * 55)
print("ch1 实验 1：动作离散化（256 桶方案）")
print("=" * 55)

# ═══ 1. 动作空间定义 ═══
# VLA 输出的是末端执行器增量：Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper
# 每个维度的值域是 [-1, 1]（OpenVLA 的设定）
ACTION_DIM = 7
NUM_BINS = 256
LOW, HIGH = -1.0, 1.0

print(f"\n动作维度: {ACTION_DIM}  (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)")
print(f"值域: [{LOW}, {HIGH}]")
print(f"桶数: {NUM_BINS}")
print(f"分辨率: {(HIGH-LOW)/(NUM_BINS-1):.5f}   (最小可分辨的变化量)")

# ═══ 2. 连续值 → 离散桶 ═══
print(f"\n{'='*55}")
print("步骤 1：连续值 → 离散桶索引")
print(f"{'='*55}")

# 模拟 VLA 输出：一个 7 维动作
action = np.array([0.35, -0.12, 0.08, 0.0, -0.05, 0.02, 1.0])
names = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]

print(f"\n原始动作: {action}")
print(f"\n{'维度':>8}  {'连续值':>8}  {'桶索引':>8}  {'恢复值':>8}  {'误差':>10}")
print("-" * 55)

for i in range(ACTION_DIM):
    val = action[i]
    # ── 核心公式：连续值 → 桶索引 ──
    bucket = int(np.clip((val - LOW) / (HIGH - LOW) * (NUM_BINS - 1), 0, NUM_BINS - 1))
    # ── 反向：桶索引 → 连续值（桶中心） ──
    recovered = LOW + bucket / (NUM_BINS - 1) * (HIGH - LOW)
    error = abs(val - recovered)
    
    print(f"{names[i]:>8}  {val:8.4f}  {bucket:8d}  {recovered:8.4f}  {error:10.5f}")

# ═══ 3. 边界测试 ═══
print(f"\n{'='*55}")
print("步骤 2：边界测试")
print(f"{'='*55}")

test_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
print(f"\n{'输入':>8} → {'桶索引':>8} → {'恢复':>8}")
print("-" * 35)
for v in test_vals:
    b = int(np.clip((v - LOW) / (HIGH - LOW) * (NUM_BINS - 1), 0, NUM_BINS - 1))
    r = LOW + b / (NUM_BINS - 1) * (HIGH - LOW)
    print(f"{v:8.2f} → {b:8d} → {r:8.4f}")

# ═══ 4. 为什么 256？ ═══
print(f"\n{'='*55}")
print("🔑 为什么是 256？")
print(f"{'='*55}")
print(f"  LLM 的词汇表大小通常是 32000/50000+ 的 token")
print(f"  每个动作维度 = 256 个桶 = 像 LLM 词汇表里的 256 个\"动作词\"")
print(f"  7 维 × 256 桶 = 总共 7×256 = {7*256} 个可能的\"动作 token\"")
print(f"")
print(f"  类比：LLM 预测\"猫\"的下一个词是\"坐\"（从 50000 词中选 1 个）")
print(f"        VLA 预测\"末端 Δx 是多少\"（从 256 桶中选 1 个）")
print(f"")
print(f"  选 256 的原因：")
print(f"    1. 2⁸ = 256 → 1 个字节刚好装下 1 个 token ID")
print(f"    2. 分辨率 {(HIGH-LOW)/(NUM_BINS-1):.4f} 对机器人控制足够（人的手也达不到更高精度）")
print(f"    3. 和 LLM tokenizer 的词汇表大小完全无关——这是单独的\"动作词汇表\"")
