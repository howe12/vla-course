#!/usr/bin/env python3
"""Step 3 实验：OpenVLA 7B + LIBERO-Spatial 零样本推理

加载 OpenVLA 7B 模型，在 LIBERO 仿真中运行完整的 VLA 推理闭环。
首次运行会自动从 HuggingFace 下载模型 (~14GB)。

用法（本地 GPU）：
    uv pip install transformers accelerate torch
    uv run python codes/step3_openvla/run_openvla_libero.py

用法（L40 云端，推荐）：
    rsync -avz -e "ssh -p 30334" codes/step3_openvla/ root@120.209.70.195:/root/gpufree-data/vla-course/codes/step3_openvla/
    ssh -p 30334 root@120.209.70.195
    cd /root/gpufree-data/vla-course && uv run python codes/step3_openvla/run_openvla_libero.py
"""

import sys
import numpy as np
from PIL import Image

# 检查依赖
try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import TASK_MAPPING
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: uv pip install transformers accelerate torch libero pillow")
    sys.exit(1)


def main():
    print("=" * 60)
    print("Step 3 实验：OpenVLA 7B + LIBERO 零样本推理")
    print("=" * 60)

    # ============================================================
    # 1. 加载 OpenVLA 模型
    # ============================================================
    import os
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 优先使用本地模型路径（L40 云端 / VLA-L40 容器）
    LOCAL_MODEL_PATH = "/root/gpufree-data/models/openvla-7b"
    if os.path.isdir(LOCAL_MODEL_PATH):
        MODEL_ID = LOCAL_MODEL_PATH
    else:
        MODEL_ID = "openvla/openvla-7b"

    print(f"\n[1/4] 加载 OpenVLA 模型到 {DEVICE}...")
    print(f"  模型: {MODEL_ID}")
    if MODEL_ID == "openvla/openvla-7b":
        print(f"  首次运行会自动下载 ~14GB，请耐心等待...")

    if DEVICE == "cpu":
        print("  ⚠️  未检测到 GPU，将使用 CPU 推理（速度较慢）")

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("\n可能原因：")
        print("  1. 网络无法访问 HuggingFace → 设置 HF_ENDPOINT=https://hf-mirror.com")
        print("  2. 显存不足 → 尝试 INT4 量化（见 Step 3.3 课程内容）")
        print("  3. 磁盘空间不足 → 需要 ~14GB")
        sys.exit(1)

    print(f"  ✅ 模型加载完成")

    # ============================================================
    # 2. 加载 LIBERO 任务
    # ============================================================
    print("\n[2/4] 加载 LIBERO-Spatial 任务...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(0)

    bddl_dir = os.path.join(get_libero_path("bddl_files"), "libero_spatial")
    bddl_path = os.path.join(bddl_dir, task.bddl_file)

    env = TASK_MAPPING["libero_tabletop_manipulation"](
        bddl_file_name=bddl_path,
        robots=["Panda"],
        has_offscreen_renderer=True,
        has_renderer=False,
        camera_heights=224,
        camera_widths=224,
        render_gpu_device_id=-1,
    )

    print(f"  任务: {task.name}")
    print(f"  指令: {task.language}")

    # ============================================================
    # 3. 推理循环
    # ============================================================
    print("\n[3/4] 开始推理循环...")

    obs = env.reset()
    max_steps = 30
    success = False

    for step in range(max_steps):
        # --- 图像预处理 ---
        image = Image.fromarray(obs["agentview_image"]).convert("RGB")

        # --- 构建 OpenVLA Prompt ---
        instruction = task.language
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # --- 模型推理 ---
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)

        with torch.no_grad():
            # 使用 token 解码方式（兼容所有 unnorm_key）
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False,
            )

        # --- Token → 连续动作 ---
        NUM_BINS = 256
        BIN_CENTERS = np.linspace(-1.0, 1.0, NUM_BINS)
        new_tokens = generated_ids[0, inputs["input_ids"].shape[1] :]
        action = np.array(
            [BIN_CENTERS[min(new_tokens[i].item(), NUM_BINS - 1)] for i in range(7)]
        )

        # 补 1 维 (保留位) → 8 维
        action_8d = np.append(action, [0.0])
        # --- 执行动作 ---
        obs, reward, done, info = env.step(action_8d)

        if step % 5 == 0 or done:
            a = action
            print(
                f"  Step {step:2d} | "
                f"Δxyz=({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f}) | "
                f"Δrpy=({a[3]:+.3f},{a[4]:+.3f},{a[5]:+.3f}) | "
                f"gripper={a[6]:+.3f}"
            )

        if done:
            success = True
            print(f"\n  ✅ 任务完成于 step {step}！")
            break

    # ============================================================
    # 4. 结果
    # ============================================================
    print("\n[4/4] 推理结果")
    print(f"  成功率: {'✅ 成功' if success else '❌ 未完成'} (1 个任务)")
    if not success:
        print(f"  模型在 {max_steps} 步内未完成任务。")
        print(f"  提示: 运行全部 10 个 LIBERO-Spatial 任务来统计真正的成功率")

    print(f"\n  ⏭️  下一步: Step 5 — 训练你自己的 SmolVLA 模型")

    env.close()


if __name__ == "__main__":
    main()
