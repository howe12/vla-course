#!/usr/bin/env python3
"""Task 01: AMD ROCm 设备与环境确认

来源: Datawhale 01_device_env_check.ipynb
用法: python3 dw_reference/01_check_env.py
前置: source /workspace/venv/bin/activate
"""

import subprocess, torch, json, os, sys
from pathlib import Path

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

def main():
    print("=" * 60)
    print("Task 01: AMD ROCm 设备与环境确认")
    print("=" * 60)

    # --- Checkpoint 1: ROCm + PyTorch ---
    print("\n>>> Checkpoint 1: ROCm 和 PyTorch 能否看到 GPU\n")
    
    product = run("rocm-smi --showproductname 2>&1")
    vram = run("rocm-smi --showmeminfo vram 2>&1")
    rocm_ver = Path("/opt/rocm/.info/version").read_text().strip() if Path("/opt/rocm/.info/version").exists() else "unknown"
    
    print("ROCm 版本:", rocm_ver)
    print("GPU 产品名:", [l for l in product.split('\n') if 'Card Series' in l or 'GPU[' in l])
    print()

    try:
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        print(f"GPU: {gpu_name} ({vram_gb} GB)")
        cuda_ok = torch.cuda.is_available()
    except Exception as e:
        print(f"❌ PyTorch CUDA 检查失败: {e}")
        cuda_ok = False

    if cuda_ok:
        print("\n✅ ROCm → PyTorch 链路已打通")
    else:
        print("\n❌ 请检查 PyTorch 是否安装 ROCm 版本")
        sys.exit(1)

    # --- Checkpoint 2: 磁盘目录 ---
    print("\n>>> Checkpoint 2: 磁盘与目录\n")
    
    paths = {
        "workspace": "/workspace",
        "models": "/workspace/vla-course/models",
        "root": "/",
    }
    for name, p in paths.items():
        try:
            stat = os.statvfs(p)
            free_gb = round(stat.f_frsize * stat.f_bavail / 1024**3, 1)
            total_gb = round(stat.f_frsize * stat.f_blocks / 1024**3, 1)
            print(f"  {name:15s}  可用 {free_gb:>8.1f} GB / 总计 {total_gb:>8.1f} GB")
        except:
            print(f"  {name:15s}  (不可访问)")

    # --- 设备资源表 ---
    print("\n>>> 设备资源表\n")
    
    info = {
        "GPU 型号": torch.cuda.get_device_name(0),
        "显存 (GB)": vram_gb,
        "ROCm 版本": rocm_ver,
        "PyTorch": torch.__version__,
        "CUDA 可用": torch.cuda.is_available(),
        "Python": sys.version.split()[0],
    }
    for k, v in info.items():
        print(f"  {k:20s} {v}")

    # 保存 JSON
    out_path = Path("/workspace/device_info.json")
    out_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\n设备信息已保存: {out_path}")

    print("\n✅ Task 01 完成。环境就绪，可以继续 Task 07 数据采集或 Task 11 零训练预览。")

if __name__ == "__main__":
    main()
