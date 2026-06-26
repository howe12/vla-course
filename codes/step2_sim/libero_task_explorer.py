#!/usr/bin/env python3
"""LIBERO 入门 1/2：任务浏览器 — 理解 LIBERO 基准测试的结构

探索 LIBERO 的 4 个子集、130 个任务、每个任务的指令和属性。
这是理解"VLA 考什么"的第一步。

运行：
    cd /develop/vla-course/codes
    source .venv/bin/activate
    python step2_sim/libero_task_explorer.py
"""

import os
from libero.libero import benchmark

print("=" * 55)
print("LIBERO 入门 1/2：任务浏览器")
print("=" * 55)

# ═══ 1. 基准结构 ═══
benchmark_dict = benchmark.get_benchmark_dict()
print(f"\n📊 LIBERO 包含 {len(benchmark_dict)} 个子集:\n")

total_tasks = 0
for name, suite_fn in benchmark_dict.items():
    try:
        suite = suite_fn()
        n = suite.n_tasks
    except Exception:
        n = "?"
    total_tasks += n if isinstance(n, int) else 0
    desc = {
        "libero_spatial": "空间推理（关系理解，10 任务）",
        "libero_object":  "物体识别（颜色/形状，10 任务）",
        "libero_goal":    "目标导向（多步操作，10 任务）",
        "libero_100":     "综合评测（100 任务全集）",
        "libero_90":      "预训练集（LIBERO-100 去重，90 任务）",
        "libero_10":      "长时程任务（10 步+ 操作，10 任务）",
    }.get(name, f"{n} 个任务")

    print(f"  📦 {name:20s} → {str(n):>3s} 个任务 | {desc}")

print(f"\n  总计: {total_tasks} 个任务 (含重叠)")

# ═══ 2. 深入 LIBERO-Spatial ═══
print(f"\n{'='*55}")
print("🔍 深入 LIBERO-Spatial（最常用的 VLA 零样本评测集）")
print(f"{'='*55}")

suite = benchmark_dict["libero_spatial"]()
print(f"\n  共 {suite.n_tasks} 个任务:\n")

for i in range(suite.n_tasks):
    task = suite.get_task(i)
    bddl = task.bddl_file.replace(".bddl", "")
    # 截断过长的名字
    short = bddl[:50] + "..." if len(bddl) > 50 else bddl
    print(f"  [{i}] {short}")
    print(f"      指令: {task.language}")
    print()

# ═══ 3. 任务属性 ═══
print(f"{'='*55}")
print("🔍 单个任务的完整属性 (以任务 0 为例)")
print(f"{'='*55}")

task0 = suite.get_task(0)
print(f"""
  task.name:                  {task0.name}
  task.language:              {task0.language}
  task.bddl_file:             {task0.bddl_file}
  task.problem:               {task0.problem}
  task.problem_folder:        {task0.problem_folder}

  💡 BDDL = Behavior Domain Definition Language
     描述"场景中有什么 + 任务目标是什么"的规范文件
""")

# ═══ 4. 环境创建（只创建不运行） ═══
print(f"{'='*55}")
print("🔍 环境创建（理解 env_args）")
print(f"{'='*55}")

from libero.libero.envs import TASK_MAPPING
from libero.libero import get_libero_path

bddl_dir = os.path.join(get_libero_path("bddl_files"), "libero_spatial")
bddl_path = os.path.join(bddl_dir, task0.bddl_file)

env = TASK_MAPPING["libero_tabletop_manipulation"](
    bddl_file_name=bddl_path,
    robots=["Panda"],
    has_offscreen_renderer=True,
    has_renderer=False,
    camera_heights=224,
    camera_widths=224,
    render_gpu_device_id=-1,
)

obs = env.reset()

print(f"\n  观测空间: {len(obs)} 个 key")
print(f"  图像类:")
for k, v in obs.items():
    import numpy as np
    if isinstance(v, np.ndarray) and v.ndim == 3:
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min()},{v.max()}]")

print(f"\n  本体感知类 (proprioception):")
for k, v in obs.items():
    import numpy as np
    if isinstance(v, np.ndarray) and v.ndim == 1 and 'image' not in k:
        print(f"    {k}: dim={len(v)}")

print(f"\n  动作空间: {env.action_spec[0].shape[0]} 维")
print(f"    前 7 维 = 末端增量 (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)")
print(f"    第 8 维 = gripper 控制")

env.close()
print(f"\n✅ 任务浏览器完成！")
print(f"   ⏭️  下一步: libero_action_test.py — 理解动作空间")
