#!/usr/bin/env python3
"""Task 07: LeRobot 数据集审计

来源: Datawhale 07_data_collection_and_audit.ipynb（审计部分）
前置: source /workspace/venv/bin/activate

用法:
  # 审计真实采集的数据集（需要完整的 LeRobot 数据集目录）
  python3 dw_reference/07_audit_data.py --dataset /workspace/datasets/my_cup_pick

  # 使用 Datawhale 提供的示例数据预览审计流程（无需真实数据集）
  python3 dw_reference/07_audit_data.py --demo

审计三步骤：
    1️⃣ 数据量 — episode 数、总帧数、任务分布
    2️⃣ 视频回放 — 提示检查关键帧（需手动观看视频）
    3️⃣ 动作范围 — action/state 统计与异常检测

注意：键盘遥操作采集本身不可脚本化（需 GUI），本脚本只做采集后的数据审计。
"""

import argparse, json, os, sys
from pathlib import Path

# ── 工具函数 ───────────────────────────────────────────

def load_json(path: str):
    """加载 JSON 文件，文件不存在返回 None"""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def load_jsonl(path: str):
    """加载 JSONL 文件，每行一个 JSON"""
    p = Path(path)
    if not p.exists():
        return None
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def fmt(num):
    """格式化数字"""
    if isinstance(num, float):
        return f"{num:.3f}"
    return str(num)

# ── 各审计步骤 ─────────────────────────────────────────

def audit_quantity(dataset_path: str):
    """步骤 1️⃣：数据量审计"""
    print("\n" + "=" * 60)
    print("1️⃣  数据量审计")
    print("=" * 60)

    info = load_json(f"{dataset_path}/meta/info.json")
    stats = load_json(f"{dataset_path}/meta/stats.json")
    episodes = load_jsonl(f"{dataset_path}/meta/episodes.jsonl")

    if info:
        print(f"\n📋 info.json")
        fps = info.get("fps", "?")
        features = info.get("features", {})
        print(f"   帧率 (fps): {fps}")
        if features:
            for k, v in features.items():
                print(f"   {k}: shape={v.get('shape','?')}, dtype={v.get('dtype','?')}")

    if stats:
        print(f"\n📊 stats.json — action/observation 统计")
        for key in ["action", "observation.state"]:
            if key in stats:
                s = stats[key]
                print(f"   {key}:")
                print(f"     min  = [{', '.join(fmt(x) for x in s['min'])}]")
                print(f"     max  = [{', '.join(fmt(x) for x in s['max'])}]")
                print(f"     mean = [{', '.join(fmt(x) for x in s['mean'])}]")

    if episodes:
        lengths = [ep["length"] for ep in episodes]
        total_frames = sum(lengths)
        print(f"\n📦 episodes.jsonl")
        print(f"   总 episode 数: {len(episodes)}")
        print(f"   总帧数:        {total_frames}")
        print(f"   平均帧数:      {total_frames / len(episodes):.0f}")
        print(f"   最短/最长:     {min(lengths)} / {max(lengths)} 帧")

        # 任务分布（如果 episode 有 task/language_instruction 字段）
        tasks = {}
        for ep in episodes:
            task = ep.get("task", ep.get("language_instruction", "未标注"))
            # 简化红/蓝杯判断
            if "red" in str(task).lower() or "红" in str(task):
                task_key = "🔴 红杯"
            elif "blue" in str(task).lower() or "蓝" in str(task):
                task_key = "🔵 蓝杯"
            else:
                task_key = str(task)[:40]
            tasks[task_key] = tasks.get(task_key, 0) + 1

        if tasks:
            print(f"\n📊 任务分布:")
            for t, n in sorted(tasks.items(), key=lambda x: -x[1]):
                pct = n / len(episodes) * 100
                print(f"   {t}: {n} ({pct:.0f}%)")

        # 帧长分布
        print(f"\n📏 帧长分布:")
        buckets = {}
        for l in lengths:
            bucket = (l // 50) * 50
            buckets[bucket] = buckets.get(bucket, 0) + 1
        for b in sorted(buckets):
            print(f"   {b}-{b+49} 帧: {buckets[b]} episodes")
    else:
        print("\n⚠️ 未找到 episodes.jsonl，跳过数据量统计")

    return {"total_episodes": len(episodes) if episodes else 0, "total_frames": total_frames if episodes else 0}

def audit_video(dataset_path: str):
    """步骤 2️⃣：视频回放提示"""
    print("\n" + "=" * 60)
    print("2️⃣  视频回放审计（手动）")
    print("=" * 60)

    video_dir = Path(dataset_path) / "videos"
    mp4_files = list(video_dir.glob("*.mp4")) if video_dir.exists() else []

    if mp4_files:
        print(f"\n🎬 找到 {len(mp4_files)} 个视频文件")
        for vf in sorted(mp4_files)[:5]:
            size_mb = vf.stat().st_size / 1024 / 1024
            print(f"   {vf.name} ({size_mb:.1f} MB)")
        if len(mp4_files) > 5:
            print(f"   ... 还有 {len(mp4_files) - 5} 个")

        print(f"""
📺 请手动逐 episode 观看视频，检查以下四问：
   1. 机械臂是否接触了杯子？
   2. 夹爪是否夹起了杯子？
   3. 杯子是否被搬运到了目标位置？
   4. 杯子放置后是否保持直立？

   视频路径: {video_dir.resolve()}
   下载到本地后使用任意播放器观看。
""")
    else:
        print(f"\n⚠️ 未找到视频目录: {video_dir}")
        print("   采集数据时需开启视频录制以支持视觉审计。")

def audit_actions(dataset_path: str):
    """步骤 3️⃣：动作范围检查"""
    print("\n" + "=" * 60)
    print("3️⃣  动作 / 状态范围检查")
    print("=" * 60)

    data_dir = Path(dataset_path) / "data"
    if not data_dir.exists():
        print(f"\n⚠️ 数据目录不存在: {data_dir}")
        return

    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        print(f"\n⚠️ 未找到 parquet 文件")
        return

    print(f"\n🔍 扫描 {len(parquet_files)} 个 parquet 文件...")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("\n⚠️ pyarrow 未安装，跳过 parquet 深度检查。")
        print("   安装: pip install pyarrow")
        return

    all_actions = []
    bad_files = []

    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
            df = table.to_pandas()

            if "action" in df.columns:
                # 取每帧 action 向量
                for _, row in df.iterrows():
                    act = row["action"]
                    if hasattr(act, "__len__"):
                        all_actions.append(list(act))
                    else:
                        all_actions.append([act])

            # 检查帧数
            n_frames = len(df)
            if n_frames == 0:
                bad_files.append((pf.name, "空文件"))

        except Exception as e:
            bad_files.append((pf.name, str(e)))

    if bad_files:
        print(f"\n❌ 异常文件 ({len(bad_files)}):")
        for fname, reason in bad_files:
            print(f"   {fname}: {reason}")

    if all_actions:
        n_dims = len(all_actions[0])
        print(f"\n📐 Action 维度: {n_dims}")
        print(f"   总帧数: {len(all_actions)}")
        for d in range(n_dims):
            vals = [a[d] for a in all_actions]
            print(f"   dim[{d}]: min={min(vals):.4f}  max={max(vals):.4f}  range={max(vals)-min(vals):.4f}")

        # 检查异常值
        for d in range(n_dims):
            vals = [a[d] for a in all_actions]
            rng = max(vals) - min(vals)
            if rng < 0.001:
                print(f"   ⚠️  dim[{d}] 范围极小 ({rng:.5f})，可能采集数据有问题")
            if max(abs(min(vals)), abs(max(vals))) > 100:
                print(f"   ⚠️  dim[{d}] 值域异常大 [{min(vals):.1f}, {max(vals):.1f}]")

    print("\n✅ 动作范围检查完成")

def demo_audit():
    """使用 Datawhale 提供的示例数据预览审计流程"""
    script_dir = Path(__file__).resolve().parent
    snapshot_path = script_dir / "assets" / "collection_dataset_snapshot.json"
    
    snap = load_json(str(snapshot_path))
    if not snap:
        print(f"❌ 未找到示例数据: {snapshot_path}")
        print("   请先运行: bash dw_reference/setup_dw.sh")
        sys.exit(1)

    print("╔" + "═" * 58 + "╗")
    print("║  LeRobot 数据集审计 — 演示模式（Datawhale 示例数据）   ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  repo: {snap.get('dataset_repo_id', '?'):<46s} ║")
    print("╚" + "═" * 58 + "╝")

    # ── 1️⃣ 数据量 ──
    print("\n" + "=" * 60)
    print("1️⃣  数据量审计")
    print("=" * 60)

    fps = snap.get("fps", "?")
    n_episodes = snap.get("total_episodes", 0)
    n_frames = snap.get("total_frames", 0)
    state_shape = snap.get("state_shape", [])
    action_shape = snap.get("action_shape", [])
    cameras = snap.get("stored_cameras", [])
    tasks = snap.get("tasks", [])

    print(f"\n📋 数据集概览")
    print(f"   帧率 (fps):     {fps}")
    print(f"   State 维度:     {state_shape}")
    print(f"   Action 维度:    {action_shape}")
    print(f"   相机通道:       {', '.join(cameras) if cameras else '无'}")
    print(f"\n📦 采集统计")
    print(f"   总 episode 数:  {n_episodes}")
    print(f"   总帧数:         {n_frames}")
    print(f"   平均帧数:       {n_frames / n_episodes:.0f}" if n_episodes else "   平均帧数:       N/A")

    if tasks:
        print(f"\n📊 任务列表:")
        for t in tasks:
            color = "🔴" if "red" in t.lower() else "🔵" if "blue" in t.lower() else "⚪"
            print(f"   {color} {t}")

    # ── 2️⃣ 视频回放 ──
    print("\n" + "=" * 60)
    print("2️⃣  视频回放审计（手动）")
    print("=" * 60)

    views = snap.get("recorder_views", [])
    if views:
        print(f"\n🎬 Datawhale 录制的四视角视频:")
        for v in views:
            print(f"   📹 {v}")
        print(f"\n📺 Datawhale 仓库中有对应的四视角视频文件 (.mp4)，")
        print(f"   可在 GitHub 上查看:")
        print(f"   https://github.com/datawhalechina/every-embodied/tree/main/")
        print(f"   16-专题组队学习/04-AMD-ROCm策略复刻专题/assets/")
        print(f"\n   四问检查：接触 → 夹起 → 搬运 → 直立")

    # ── 3️⃣ 动作范围 ──
    print("\n" + "=" * 60)
    print("3️⃣  动作 / 状态范围检查")
    print("=" * 60)
    print(f"\n📐 State 维度: {state_shape[0] if state_shape else '?'}")
    print(f"📐 Action 维度: {action_shape[0] if action_shape else '?'}")
    print(f"\n💡 演示模式仅展示数据集结构，不做实际 parquet 扫描。")
    print(f"   真实审计时（--dataset 模式）会逐文件读取 action 值域。")

    return {"total_episodes": n_episodes, "total_frames": n_frames}


# ── 主入口 ─────────────────────────────────────────────

def print_summary(q: dict, source: str = ""):
    """打印审计总结"""
    label = f" ({source})" if source else ""
    print("\n" + "=" * 60)
    print(f"📋 审计总结{label}")
    print("=" * 60)
    print(f"""
   总 episodes:  {q['total_episodes']}
   总帧数:       {q['total_frames']}

   下一步:
   ✅ 数据量 OK  → 进入 Ch3 物理成功评估
   ✅ 视频 OK    → 确认夹取行为正确
   ✅ 动作 OK    → 数据可用于 Ch4 训练
""")


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot 数据集三步骤审计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 审计本地采集的数据集
  python3 dw_reference/07_audit_data.py --dataset /workspace/datasets/my_cup_pick

  # 使用 Datawhale 示例数据预览（无需真实数据集）
  python3 dw_reference/07_audit_data.py --demo
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str,
                       help="数据集路径（包含 meta/ data/ videos/ 的目录）")
    group.add_argument("--demo", action="store_true",
                       help="使用 Datawhale 提供的示例数据预览审计流程")
    args = parser.parse_args()

    # ── 演示模式 ──
    if args.demo:
        q = demo_audit()
        print_summary(q, source="Datawhale 示例数据 (demo)")
        return

    # ── 真实数据集审计 ──
    ds = args.dataset
    if not os.path.isdir(ds):
        print(f"❌ 数据集路径不存在: {ds}")
        sys.exit(1)

    print("╔" + "═" * 58 + "╗")
    print("║  LeRobot 数据集审计 — 三步骤                             ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  数据集: {ds:<47s} ║")
    print("╚" + "═" * 58 + "╝")

    # 步骤 1️⃣
    q = audit_quantity(ds)

    # 步骤 2️⃣
    audit_video(ds)

    # 步骤 3️⃣
    audit_actions(ds)

    # ── 总结 ──
    print_summary(q)


if __name__ == "__main__":
    main()
