#!/usr/bin/env python3
"""Task 04: SmolVLA 迁移与采样加权数据展示

来源: Datawhale 04_smolvla_weighted_sampling.ipynb
用法: python3 dw_reference/04_smolvla_weighted.py
前置: assets/metrics_snapshot.json
"""

import json
from pathlib import Path

ASSET_DIR = Path(__file__).parent / "assets"

def main():
    print("=" * 60)
    print("Task 04: SmolVLA 红蓝杯任务偏置分析")
    print("=" * 60)

    snapshot_path = ASSET_DIR / "metrics_snapshot.json"
    if not snapshot_path.exists():
        print(f"\n❌ 缺少指标快照: {snapshot_path}")
        return

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    print("\n>>> SmolVLA 红/蓝杯成功率对照\n")
    print("  配置             红杯成功率    蓝杯成功率    说明")
    print("  ───────────────  ────────────  ────────────  ───────────────────")
    print("  baseline          8/10 (80%)    0/10  (0%)   蓝杯完全不会")
    print("  copy 1.5×         3/10 (30%)    7/10 (70%)   简单复制反效果")
    print("  copy 2×           4/10 (40%)    8/10 (80%)   ")
    print("  copy 3×           2/10 (20%)    8/10 (80%)   红杯退化")
    print("  weighted 500 ✅    8/10 (80%)   10/10 (100%)  ★ 最佳：蓝杯拉满")
    print("  weighted 1k        6/10 (60%)    9/10 (90%)   步数过多反而退化")
    print()

    print(">>> 关键概念\n")
    print("  为什么蓝杯 baseline 是 0%？")
    print("  → 数据集中蓝杯样本远少于红杯（任务偏置）")
    print("  → SmolVLA 是多模态模型，能听懂指令")
    print('  → 倾向于"忽略"蓝杯指令')
    print()
    print("  为什么 weighted 500 最好？")
    print("  → weighted sampler：蓝杯样本采样概率 = 红杯的 2 倍")
    print("  → 不是简单复制 episode，而是改变 DataLoader 采样分布")
    print("  → 500 steps 刚好：太少没学到，太多会过拟合蓝杯（weighted 1k 退化）")
    print()

    print(">>> 实操：在你的数据上做加权采样\n")
    print("  1. 采集 20-50 条数据，统计红/蓝杯比例")
    print("  2. 如果蓝杯 < 40%，在训练配置中加 weighted sampling")
    print("  3. 对比 baseline vs weighted 的 closed-loop 成功率")
    print()

    print("✅ Task 04 完成。继续 Task 07 采集数据，或 Task 08 训练 ACT。")

if __name__ == "__main__":
    main()
