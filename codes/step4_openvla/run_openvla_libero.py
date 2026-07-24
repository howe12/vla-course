#!/usr/bin/env python3
"""Step 3 实验：OpenVLA 7B + LIBERO-Spatial 微调模型推理（官方对齐版）

严格对齐 OpenVLA 官方 run_libero_eval.py：
- OffScreenRenderEnv（OSC_POSE 控制器）
- 图像 180° 旋转 + JPEG 编解码（匹配训练预处理）
- 夹爪取反 + 二值化
- 10 步初始等待
- 220 步最大推理步数
- 每任务生成截图

用法：
    cd /root/gpufree-data/vla-course/codes
    .venv/bin/python step3_openvla/run_openvla_libero.py
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
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    sys.exit(1)

# ─── 配置 ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 220            # libero_spatial: 最长 demo = 193 步
NUM_WAIT_STEPS = 10        # 初始等待帧（物体稳定）
SCREENSHOT_STEPS = [0, 40, 80, 120, 160, 200]  # 固定截帧点
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def load_model():
    """加载 OpenVLA 7B 模型和处理器"""
    LOCAL_MODEL_PATH = "/root/gpufree-data/models/openvla-7b-finetuned-libero-spatial"
    MODEL_ID = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else "openvla/openvla-7b-finetuned-libero-spatial"

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


def get_image(obs, resize_size=224, center_crop=False):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 180° 旋转（对齐官方预处理）
    # JPEG 编解码（对齐 RLDS 数据管道）
    import io
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=95)
    buf.seek(0)
    img = np.array(Image.open(buf))
    if resize_size:
        img = np.array(Image.fromarray(img).resize((resize_size, resize_size), Image.LANCZOS))
    return Image.fromarray(img).convert("RGB")
def normalize_gripper(action, binarize=True):
    """
    夹爪动作规范化：OpenVLA 输出 [0,1] → LIBERO 期望 [-1,+1]
    官方: 0=关, 1=开 → LIBERO: -1=开, +1=关
    """
    # 取反: OpenVLA (0=close,1=open) → LIBERO convention
    # LIBERO: -1=open, +1=close
    # So: flip sign
    gripper = action[..., -1]
    if binarize:
        gripper = np.where(gripper > 0, 1.0, -1.0)
    # Invert: OpenVLA close→open → LIBERO open→close
    gripper = -gripper
    if isinstance(action, np.ndarray):
        action[..., -1] = gripper
    return action


def save_screenshot(obs, task_id, step, resize=True):
    """保存观测截图"""
    img = obs["agentview_image"]
    if resize:
        img = np.array(Image.fromarray(img).resize((448, 448), Image.LANCZOS))
    path = os.path.join(SCREENSHOT_DIR, f"task{task_id:02d}_step{step:03d}.png")
    Image.fromarray(img).save(path)
    return path


def run_task(model, processor, task, task_id):
    """在单个任务上运行推理闭环（对齐官方评估管道）"""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)

    obs = env.reset()
    screenshots = {}
    success = False

    # 初始截图
    if 0 in SCREENSHOT_STEPS:
        screenshots[0] = save_screenshot(obs, task_id, 0)

    # ─── 初始等待帧（物体稳定） ──────────────────────────────
    dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    for _ in range(NUM_WAIT_STEPS):
        obs, _, _, _ = env.step(dummy_action)

    # ─── 推理循环 ────────────────────────────────────────────
    prompt = f"In: What action should the robot take to {task_description}?\nOut:"

    for step in range(MAX_STEPS):
        image = get_image(obs, resize_size=224, center_crop=True)

        inputs = processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)

        with torch.no_grad():
            action = model.predict_action(
                **inputs,
                unnorm_key="libero_spatial",  # 微调模型自带 libero_spatial norm_stats
                do_sample=False,
            )

        # 夹爪规范化：取反 + 二值化
        action = normalize_gripper(action, binarize=True)


        obs, reward, done, info = env.step(action.tolist())

        # LIBERO 成功检测
        if step > 0 and env.check_success():
            success = True
            screenshots["final"] = save_screenshot(obs, task_id, step)
            break

        # 截图
        if step in SCREENSHOT_STEPS:
            screenshots[step] = save_screenshot(obs, task_id, step)

    if not success and "final" not in screenshots:
        screenshots["final"] = save_screenshot(obs, task_id, MAX_STEPS - 1)

    env.close()
    return success, screenshots


def main():
    print("=" * 60)
    print("Step 3 实验：OpenVLA 7B + LIBERO-Spatial 微调模型推理")
    print("         10 任务评估 + 截图（官方管道对齐）")
    print("=" * 60)

    # ─── 加载模型 ─────────────────────────────────────────────
    model, processor = load_model()

    # ─── 加载任务列表 ─────────────────────────────────────────
    print("[2/3] 加载 LIBERO-Spatial 任务列表...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    num_tasks = task_suite.n_tasks
    print(f"  共 {num_tasks} 个任务（每任务最多 {MAX_STEPS} 步）\n")

    # ─── 遍历评估 ─────────────────────────────────────────────
    print(f"[3/3] 开始推理评估...")
    print(f"  截图保存: {SCREENSHOT_DIR}/")
    print(f"  等待帧: {NUM_WAIT_STEPS} | 最大步: {MAX_STEPS}")
    print()

    results = []
    t_start = time.time()

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        instruction = task.language[:80]

        print(f"  ── Task {task_id} ──────────────────────────────")
        print(f"  指令: {instruction}")

        success, screenshots = run_task(model, processor, task, task_id)

        status = "✅" if success else "❌"
        shots = len(screenshots)
        results.append((task_id, success, task.name[:60]))

        print(f"  结果: {status}  |  截图: {shots} 张")
        print()

    elapsed = time.time() - t_start
    successes = sum(r[1] for r in results)

    # ─── 汇总 ─────────────────────────────────────────────────
    print("=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"  总任务数: {num_tasks}")
    print(f"  成功:     {successes}")
    print(f"  失败:     {num_tasks - successes}")
    print(f"  成功率:   {successes/num_tasks*100:.0f}%")
    print(f"  总耗时:   {elapsed:.0f}s")
    print(f"\n  截图目录: {SCREENSHOT_DIR}/")

    print(f"\n  ⏭️  下一步: 将截图嵌入 ch3 课程 → push → GitHub Pages")


if __name__ == "__main__":
    main()
