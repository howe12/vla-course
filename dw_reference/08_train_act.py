#!/usr/bin/env python3
"""Task 08: ACT 策略训练

来源: Datawhale 08_act_training.ipynb + 16_act_end_to_end.ipynb
前置:
    source /workspace/venv/bin/activate
    pip install 'lerobot[training]'

用法:
    # Smoke test — 2 步验证数据加载+模型前向+反向传播+checkpoint 写出
    python3 dw_reference/08_train_act.py --dataset /workspace/datasets/my_data --smoke

    # 完整训练 — 需要显式设置环境变量
    RUN_REAL_TRAIN=1 python3 dw_reference/08_train_act.py --dataset /workspace/datasets/my_data --train

ACT 关键信息:
    参数量: ~30M (ResNet-18 backbone + Transformer encoder/decoder)
    基线步数: 5,000 steps (Datawhale ACT baseline)
    Datawhale 基准: 17/30 physical_success (56.7%)
"""

import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import yaml

# ── 默认配置 ──────────────────────────────────────────

ACT_DEFAULTS = {
    "seed": 1000,
    "batch_size": 8,
    "steps": 5000,               # ACT 基线
    "save_freq": 1000,
    "log_freq": 100,
    "num_workers": 4,
    "chunk_size": 100,           # action chunk 长度
    "n_action_steps": 100,
    "optimizer_lr": 1e-5,
    "optimizer_weight_decay": 1e-4,
    "vision_backbone": "resnet18",
    "dim_model": 512,
    "n_heads": 8,
    "dim_feedforward": 3200,
    "n_encoder_layers": 4,
    "n_decoder_layers": 1,
    "dropout": 0.1,
    "use_amp": False,            # AMD ROCm mixed-precision 待验证
}


# ── 环境检查 ──────────────────────────────────────────

def check_env():
    """检查 GPU、LeRobot、PyTorch 是否就绪"""
    print("\n>>> 环境检查\n")

    # GPU
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        if gpu_ok:
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            print(f"   ✅ GPU: {name} ({vram} GB)")
        else:
            print("   ❌ CUDA/ROCm 不可用。请检查 PyTorch 是否为 ROCm 版本。")
            sys.exit(1)
    except ImportError:
        print("   ❌ PyTorch 未安装")
        sys.exit(1)

    # LeRobot
    try:
        import lerobot
        print(f"   ✅ LeRobot {lerobot.__version__}")
    except ImportError:
        print("   ❌ LeRobot 未安装。执行: pip install lerobot")
        sys.exit(1)

    # 训练依赖
    try:
        from lerobot.scripts.lerobot_train import train  # noqa: F401
        print("   ✅ lerobot[training] 已安装")
    except ImportError:
        print("   ⚠️  lerobot[training] 未完整安装，尝试: pip install 'lerobot[training]'")

    print()


# ── 生成训练配置 ──────────────────────────────────────

def build_config(args):
    """根据参数构建 LeRobot TrainPipelineConfig"""
    ds_path = Path(args.dataset).resolve()
    if not ds_path.exists():
        print(f"❌ 数据集路径不存在: {ds_path}")
        sys.exit(1)

    # 尝试读取 dataset meta 获取 info
    info_path = ds_path / "meta" / "info.json"
    fps = 20
    features = {}
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        fps = info.get("fps", 20)
        features = info.get("features", {})
        print(f"   📋 数据集: {info.get('robot_type', ds_path.name)}")
        print(f"   📋 帧率: {fps} fps")
        print(f"   📋 特征: {list(features.keys())}")

    config = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.smoke_steps if args.smoke else args.steps,
        "save_freq": args.save_freq,
        "log_freq": args.log_freq,
        "num_workers": args.num_workers,
        "output_dir": str(args.output_dir),
        "job_name": args.name,
        "save_checkpoint": True,
        "save_checkpoint_to_hub": False,

        "dataset": {
            "repo_id": str(ds_path),
            "root": str(ds_path),
            "streaming": False,
        },

        "policy": {
            "type": "act",
            "push_to_hub": False,
            "chunk_size": args.chunk_size,
            "n_action_steps": args.n_action_steps,
            "vision_backbone": args.vision_backbone,
            "dim_model": args.dim_model,
            "n_heads": getattr(args, 'n_heads', ACT_DEFAULTS['n_heads']),
            "dim_feedforward": getattr(args, 'dim_feedforward', ACT_DEFAULTS['dim_feedforward']),
            "n_encoder_layers": args.n_encoder_layers,
            "n_decoder_layers": args.n_decoder_layers,
            "dropout": args.dropout,
            "use_amp": args.use_amp,
        },

        "optimizer": {
            "lr": args.optimizer_lr,
            "weight_decay": args.optimizer_weight_decay,
        },

        "wandb": {"enable": False},
        "eval": {"n_episodes": 0},
    }

    return config


# ── Smoke Test ────────────────────────────────────────

def run_smoke(config: dict):
    """2-step smoke: 验证数据加载→模型构建→前向→反向→checkpoint"""
    print("\n" + "=" * 60)
    print("🧪 Smoke Test — 2 步验证训练通路")
    print("=" * 60)

    # Save config to temp location (don't create output_dir — LeRobot will do that)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name

    print(f"\n📝 配置写入: {config_path}")

    # 运行 2-step 训练
    print(f"\n🚀 启动 smoke 训练...")
    t0 = time.time()

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--config", config_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max for smoke
        )

        elapsed = time.time() - t0

        # 检查输出
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode == 0:
            print(f"\n✅ Smoke test 通过！耗时 {elapsed:.0f}s")
            print(f"   训练脚本正常退出")

            # 检查 checkpoint 是否写出
            ckpt_dir = Path(config["output_dir"])
            ckpts = sorted(ckpt_dir.glob("*/checkpoint-*"))
            if ckpts:
                print(f"   Checkpoint 已保存: {ckpts[-1]}")
            else:
                ckpts2 = sorted(ckpt_dir.rglob("*.safetensors"))
                if ckpts2:
                    print(f"   模型文件已保存: {ckpts2[0]}")
                else:
                    print("   ⚠️  未找到 checkpoint 文件，但训练未报错")

        else:
            print(f"\n❌ Smoke test 失败 (exit code {result.returncode})")
            # 打印关键错误信息
            error_lines = []
            for line in (stderr + stdout).split("\n"):
                low = line.lower()
                if any(kw in low for kw in ["error", "traceback", "exception", "failed", "oom", "cuda"]):
                    error_lines.append(line)
            if error_lines:
                print("   关键错误:")
                for l in error_lines[-10:]:
                    print(f"   {l[:200]}")
            else:
                # 打印最后 20 行
                tail = (stderr + stdout).split("\n")[-20:]
                print("   最后输出:")
                for l in tail:
                    if l.strip():
                        print(f"   {l[:200]}")

    except subprocess.TimeoutExpired:
        print(f"\n⚠️  Smoke test 超时 (>5min)，可能卡在数据加载。检查数据集是否完整。")

    finally:
        # 保存配置文件（即使训练失败也保存，便于排查）
        final_path = Path(config["output_dir"]) / "training_config.yaml"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(config_path, final_path)
        print(f"\n📝 配置文件已保存: {final_path}")


# ── 完整训练 ──────────────────────────────────────────

def run_train(config: dict):
    """完整训练 — 需要 RUN_REAL_TRAIN=1 环境变量"""
    print("\n" + "=" * 60)
    print("🏋️  完整 ACT 训练")
    print("=" * 60)

    if os.environ.get("RUN_REAL_TRAIN") != "1":
        print("\n⚠️  完整训练需要显式确认。请设置环境变量后重新运行:")
        print(f"\n    RUN_REAL_TRAIN=1 python3 dw_reference/08_train_act.py --dataset {config['dataset']['repo_id']} --train")
        print(f"\n   预计步数: {config['steps']}")
        print(f"   预计耗时: ~{config['steps'] // 500 * 5}-{config['steps'] // 500 * 15} 分钟 (AMD GPU)")
        print(f"   输出目录: {config['output_dir']}")
        sys.exit(0)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name

    os.makedirs(config["output_dir"], exist_ok=True)

    print(f"\n📝 配置: {config_path}")
    print(f"   步数: {config['steps']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   输出目录: {config['output_dir']}")

    print(f"\n🚀 启动训练...")
    print(f"   训练日志实时输出到 stdout。中断训练用 Ctrl+C。\n")

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--config", config_path,
    ]

    result = subprocess.run(cmd)

    # 保存配置
    final_path = Path(config["output_dir"]) / "training_config.yaml"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(config_path, final_path)

    if result.returncode == 0:
        print(f"\n✅ 训练完成！")
        ckpt_dir = Path(config["output_dir"])
        ckpts = sorted(ckpt_dir.rglob("*.safetensors"))
        if ckpts:
            print(f"   Checkpoint: {ckpts[-1]}")
        print(f"\n下一步 — 闭环评估:")
        print(f"   python3 dw_reference/11_eval_closed_loop.py --checkpoint {config['output_dir']}")
    else:
        print(f"\n❌ 训练异常退出 (exit code {result.returncode})")


# ── 主入口 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ACT 策略训练 — 基于 LeRobot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Smoke test (2 步验证)
  python3 dw_reference/08_train_act.py --dataset /workspace/datasets/my_cup_data --smoke

  # 完整训练
  RUN_REAL_TRAIN=1 python3 dw_reference/08_train_act.py --dataset /workspace/datasets/my_cup_data --train

  # 自定义参数
  python3 dw_reference/08_train_act.py --dataset /workspace/datasets/my_cup_data --train --steps 10000 --batch-size 16
        """
    )

    # ── 模式选择 ──
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="2-step smoke test：验证数据加载→模型→前向→反向→checkpoint")
    mode.add_argument("--train", action="store_true",
                      help="完整训练（需 RUN_REAL_TRAIN=1 环境变量）")

    # ── 必选参数 ──
    parser.add_argument("--dataset", required=True,
                        help="LeRobot 数据集路径（本地目录或 HF repo_id）")

    # ── 训练参数 ──
    parser.add_argument("--output-dir", default="outputs/act_baseline",
                        help="输出目录 (default: outputs/act_baseline)")
    parser.add_argument("--name", default="act_baseline",
                        help="训练任务名")
    parser.add_argument("--steps", type=int, default=5000,
                        help="训练步数 (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--seed", type=int, default=1000,
                        help="随机种子 (default: 1000)")
    parser.add_argument("--save-freq", type=int, default=1000,
                        help="Checkpoint 保存频率 (default: 1000)")
    parser.add_argument("--log-freq", type=int, default=100,
                        help="日志输出频率 (default: 100)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")

    # ── ACT 模型参数 ──
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Action chunk 长度 (default: 100)")
    parser.add_argument("--n-action-steps", type=int, default=100,
                        help="执行步数 (default: 100)")
    parser.add_argument("--vision-backbone", default="resnet18",
                        help="视觉骨干网络 (default: resnet18)")
    parser.add_argument("--dim-model", type=int, default=512,
                        help="Transformer 隐层维度 (default: 512)")
    parser.add_argument("--n-encoder-layers", type=int, default=4,
                        help="Encoder 层数 (default: 4)")
    parser.add_argument("--n-decoder-layers", type=int, default=1,
                        help="Decoder 层数 (default: 1)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout (default: 0.1)")
    parser.add_argument("--optimizer-lr", type=float, default=1e-5,
                        help="学习率 (default: 1e-5)")
    parser.add_argument("--optimizer-weight-decay", type=float, default=1e-4,
                        help="权重衰减 (default: 1e-4)")
    parser.add_argument("--use-amp", action="store_true",
                        help="启用 AMP 混合精度（AMD ROCm 待验证）")

    args = parser.parse_args()

    # Smoke test 固定 2 步
    args.smoke_steps = 2

    # ── 环境检查 ──
    check_env()

    # ── 构建配置 ──
    config = build_config(args)

    # ── 运行 ──
    if args.smoke:
        run_smoke(config)
    else:
        run_train(config)


if __name__ == "__main__":
    main()
