#!/usr/bin/env python3
"""Step 3 实验：OpenVLA 7B + LIBERO-Spatial 零样本推理（10 任务评估）

加载 OpenVLA 7B 模型，在全部 10 个 LIBERO-Spatial 任务上运行推理闭环，
保存每个任务的截图并统计成功率。

用法：
    本机同步 + VLA-L40 运行:
    git push && ssh VLA-L40 "cd ... && git pull && .venv/bin/python step3_openvla/run_openvla_libero.py"
"""

import sys
import os
import time
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import TASK_MAPPING
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: uv pip install transformers accelerate torch libero pillow")
    sys.exit(1)

# ─── 配置 ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 30
SCREENSHOT_STEPS = [0, 5, 10, 20]  # 固定截帧点
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

VOCAB_SIZE = 32000
NUM_BINS = 255
BIN_CENTERS = np.linspace(-1.0, 1.0, NUM_BINS)


def load_model():
    """加载 OpenVLA 7B 模型和处理器"""
    LOCAL_MODEL_PATH = "/root/gpufree-data/models/openvla-7b"
    MODEL_ID = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else "openvla/openvla-7b"

    print(f"[1/3] 加载 OpenVLA 7B 模型...")
    print(f"  路径: {MODEL_ID}  |  设备: {DEVICE}")

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"  ✅ 模型加载完成  |  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB\n")
    return model, processor


def create_env(task):
    """根据 task 创建 LIBERO 环境"""
    bddl_dir = os.path.join(get_libero_path("bddl_files"), "libero_spatial")
    bddl_path = os.path.join(bddl_dir, task.bddl_file)
    return TASK_MAPPING["libero_tabletop_manipulation"](
        bddl_file_name=bddl_path,
        robots=["Panda"],
        has_offscreen_renderer=True,
        has_renderer=False,
        camera_heights=224,
        camera_widths=224,
        render_gpu_device_id=-1,
    )


def save_screenshot(obs, task_id, step):
    """保存观测截图"""
    img = Image.fromarray(obs["agentview_image"]).convert("RGB")
    path = os.path.join(SCREENSHOT_DIR, f"task{task_id:02d}_step{step:02d}.png")
    img.save(path)
    return path


def run_task(model, processor, task, task_id):
    """在单个任务上运行推理闭环，返回 (success, screenshots)"""
    env = create_env(task)
    obs = env.reset()
    screenshots = {}
    success = False

    # 初始截图
    if 0 in SCREENSHOT_STEPS:
        screenshots[0] = save_screenshot(obs, task_id, 0)

    instruction = task.language
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    for step in range(MAX_STEPS):
        image = Image.fromarray(obs["agentview_image"]).convert("RGB")
        inputs = processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=7, do_sample=False)

        # Token → 动作
        new_tokens = generated_ids[0, inputs["input_ids"].shape[1]:]
        discretized = VOCAB_SIZE - new_tokens.cpu().numpy().astype(np.int64) - 1
        discretized = np.clip(discretized, 0, NUM_BINS - 1)
        action = BIN_CENTERS[discretized]

        action_8d = np.append(action, [0.0])
        obs, reward, done, info = env.step(action_8d)

        # 截图
        if step in SCREENSHOT_STEPS:
            screenshots[step] = save_screenshot(obs, task_id, step)

        if done:
            success = True
            screenshots["final"] = save_screenshot(obs, task_id, step)
            break

    # 最终帧（如未 done）
    if not success and "final" not in screenshots:
        screenshots["final"] = save_screenshot(obs, task_id, MAX_STEPS - 1)

    env.close()
    return success, screenshots


def main():
    print("=" * 60)
    print("Step 3 实验：OpenVLA 7B + LIBERO-Spatial 零样本推理")
    print("         10 任务评估 + 截图")
    print("=" * 60)

    # ─── 加载模型（一次）────────────────────────────────────────
    model, processor = load_model()

    # ─── 加载任务列表 ───────────────────────────────────────────
    print("[2/3] 加载 LIBERO-Spatial 任务列表...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    num_tasks = task_suite.n_tasks
    print(f"  共 {num_tasks} 个任务\n")

    # ─── 遍历评估 ───────────────────────────────────────────────
    print(f"[3/3] 开始推理评估（每任务最多 {MAX_STEPS} 步）...")
    print(f"  截图保存: {SCREENSHOT_DIR}/")
    print()

    results = []
    t_start = time.time()

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_name = task.name[:60]
        instruction = task.language[:80]

        print(f"  ── Task {task_id} ──────────────────────────────")
        print(f"  指令: {instruction}")

        success, screenshots = run_task(model, processor, task, task_id)

        status = "✅" if success else "❌"
        shots = len(screenshots)
        results.append((task_id, success, shots, task_name))

        print(f"  结果: {status}  |  截图: {shots} 张")
        print()

    elapsed = time.time() - t_start

    # ─── 汇总 ───────────────────────────────────────────────────
    print("=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"  总任务数: {num_tasks}")
    print(f"  成功:     {sum(r[1] for r in results)}")
    print(f"  失败:     {sum(1 for r in results if not r[1])}")
    success_rate = sum(r[1] for r in results) / num_tasks * 100
    print(f"  成功率:   {success_rate:.0f}%")
    print(f"  总耗时:   {elapsed:.0f}s（约 {elapsed/num_tasks:.0f}s/任务）")
    print(f"\n  截图目录: {SCREENSHOT_DIR}/")
    print(f"  截图数量: {sum(r[2] for r in results)} 张")

    print(f"\n  ⏭️  下一步: 将截图嵌入 ch3 课程 → push → GitHub Pages")


if __name__ == "__main__":
    main()
