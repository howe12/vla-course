#!/usr/bin/env python3
"""ch1 入门实验 2/2：Token 编码解码 — 动作 ↔ Token ID

OpenVLA 把 7 维动作编码为 8 个 token ID，LLaMA 自回归生成这些 token，
然后解码回连续动作。本实验模拟这个全过程。

运行：
    cd /develop/vla-course/codes && source .venv/bin/activate
    python step1_basics/token_encode_decode.py
"""

import numpy as np

print("=" * 55)
print("ch1 实验 2：Token 编码解码")
print("=" * 55)

NUM_BINS = 256
LOW, HIGH = -1.0, 1.0

# ═══ 1. 编码：动作 → Token ID ═══
print(f"\n{'='*55}")
print("步骤 1：动作 → Token ID（编码）")
print(f"{'='*55}")

# 模拟 VLA 输出的 7 维动作
action = np.array([0.35, -0.12, 0.78, -0.05, 0.0, 0.02, 0.5])
names = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]

print(f"输入动作: {action}\n")

token_ids = []
print(f"{'维度':>8}  {'连续值':>8}  {'桶索引':>6}  {'Token ID':>10}  {'ID//256':>8}  {'ID%256':>8}")
print("-" * 65)

for i in range(7):
    val = action[i]
    bucket = int(np.clip((val - LOW) / (HIGH - LOW) * (NUM_BINS - 1), 0, NUM_BINS - 1))
    # ── 核心公式：Token ID = 维度索引 × 256 + 桶索引 ──
    token_id = i * NUM_BINS + bucket
    token_ids.append(token_id)
    
    dim_from_id = token_id // NUM_BINS  # 商 → 维度索引
    bin_from_id = token_id % NUM_BINS   # 余数 → 桶索引
    
    print(f"{names[i]:>8}  {val:8.4f}  {bucket:6d}  {token_id:10d}  {dim_from_id:8d}  {bin_from_id:8d}")

# ═══ 2. 解码：Token ID → 动作 ═══
print(f"\n{'='*55}")
print("步骤 2：Token ID → 动作（解码）")
print(f"{'='*55}")

# 模拟 LLM 自回归生成的 8 个 token ID
# （第 8 个是特殊的 EOS/停止 token，这里用 -1 示意）
generated_ids = token_ids + [-1]
print(f"\nLLM 生成了 {len(generated_ids)} 个 token: {generated_ids}")
print(f"  前 7 个 = 动作 token，第 8 个 = 停止 token\n")

recovered = np.zeros(7)
print(f"{'Token ID':>10}  {'维度':>6}  {'桶':>6}  {'恢复值':>8}")
print("-" * 40)

for tid in generated_ids[:-1]:  # 跳过停止 token
    dim = tid // NUM_BINS
    bkt = tid % NUM_BINS
    val = LOW + bkt / (NUM_BINS - 1) * (HIGH - LOW)
    recovered[dim] = val
    print(f"{tid:10d}  {dim:6d}  {bkt:6d}  {val:8.4f}")

print(f"\n恢复的动作: {np.round(recovered, 4)}")
print(f"原始的动作: {np.round(action, 4)}")
print(f"最大误差: {np.abs(action - recovered).max():.5f}")

# ═══ 3. Token ID 范围 ═══
print(f"\n{'='*55}")
print("步骤 3：Token ID 范围分析")
print(f"{'='*55}")

print(f"\n  7 个维度 × 256 桶 = 7×256 = {7*256} 个可能的动作 token")
print(f"  Token ID 范围: [0, {7*256-1}] = [0, {7*256-1}]")
print(f"  停止 token ID: -1 或 {7*256}（取决于实现）")
print(f"")
print(f"  这 1792 个 token 被插入 LLM 的 tokenizer 词汇表")
print(f"  所以 LLaMA 的词汇表从 ~32000 扩展到 ~33792")

# ═══ 4. 可视化 ═══
print(f"\n{'='*55}")
print("步骤 4：编码解码全过程可视化")
print(f"{'='*55}")
print(f"""
  原始动作 (7 维连续值)
  [0.35, -0.12, 0.78, -0.05, 0.0, 0.02, 0.5]
  
        ↓ step 1: 每维 → 256 桶索引
  
  桶索引 (7 个整数)
  [173, 112, 227, 121, 128, 130, 191]
  
        ↓ step 2: 维度×256 + 桶 → Token ID
  
  Token ID (7 个整数 + 1 个停止)
  [173, 368, 739, 889, 1152, 1410, 1727, -1]
  
        ↓ step 3: LLM 自回归生成这些 token
  
        ↓ step 4: Token ID // 256 = 维度, Token ID % 256 = 桶 → 恢复连续值
  
  恢复动作
  [0.349, -0.122, 0.780, -0.051, 0.004, 0.020, 0.498]
  
  误差 < {1/(NUM_BINS-1):.4f}  (1/255 = 离散化精度)
""")

print("🔑 整个 VLA 推理的本质：")
print("   图像+指令 → LLM 自回归 → 输出 8 个整数 → 解码为关节角 → 控制机械臂")
print("   「连续动作」→「离散 Token」→「LLM 预测」→「离散 Token」→「连续动作」")
