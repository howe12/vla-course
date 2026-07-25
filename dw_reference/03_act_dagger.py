#!/usr/bin/env python3
"""Task 03: ACT 迁移与 DAgger 诊断数据展示

来源: Datawhale 03_act_dagger_diagnostics.ipynb
用法: python3 dw_reference/03_act_dagger.py
前置: assets/metrics_snapshot.json
"""

import json
from pathlib import Path

ASSET_DIR = Path(__file__).parent / "assets"

def main():
    print("=" * 60)
    print("Task 03: ACT DAgger 诊断")
    print("=" * 60)

    snapshot_path = ASSET_DIR / "metrics_snapshot.json"
    if not snapshot_path.exists():
        print(f"\n❌ 缺少指标快照: {snapshot_path}")
        print("请先复制 Datawhale 的 assets/metrics_snapshot.json 到此目录")
        return

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    print("\n>>> ACT 进展曲线\n")
    print("Datawhale 实测 ACT 从基线到 DAgger 的进展：\n")
    print("  阶段                    physical_success   说明")
    print("  ───────────────────────  ─────────────────  ──────────────────────")
    print("  clean closed-loop        0/10  (0%)        基线：模型完全自主")
    print("  time offset              3/15  (20%)       加入时间偏移")
    print("  downweight DAgger        13/30 (43%)       DAgger + 降权")
    print("  repair15 (保护权重)       15/30 (50%)       三面板 3/10+4/10+8/10")
    print("  历史最佳 DAgger           17/30 (57%)       未验证")
    print()

    print(">>> 关键概念\n")
    print("  open-loop replay : 用录制的 action 序列回放（不依赖模型）")
    print("  closed-loop rollout: 模型每一步自主推理 action（真实能力）")
    print("  DAgger           : 人类采集 prefix 40 帧 → 模型续写 →")
    print("                     人类纠正续写结果 → 加入训练集")
    print()

    print(">>> ACT 评估命令模板\n")
    print("当你训练完 ACT 后，用以下命令做 closed-loop 评估：\n")
    print('  export POLICY_TYPE=act')
    print('  export EVAL_SEEDS="1000 1001 1002 1003 1004"')
    print('  python3 dw_reference/11_eval_closed_loop.py')
    print()
    print("  输出: dw_reference/outputs/act_closed_loop/results.jsonl")

    print("\n✅ Task 03 完成。继续 Task 04 SmolVLA 红蓝杯分析。")

if __name__ == "__main__":
    main()
