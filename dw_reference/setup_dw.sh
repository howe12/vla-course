#!/usr/bin/env bash
# dw_reference/setup_dw.sh — Datawhale 实操脚本环境准备
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Datawhale 实操环境准备 ==="
echo ""

# Check venv
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "⚠️  未检测到虚拟环境，请先执行:"
    echo "   source /workspace/venv/bin/activate"
    exit 1
fi

# Install optional deps for notebook execution
echo ">>> 安装可选依赖 (jupyter/nbclient)"
pip install -q jupyter nbclient pyyaml 2>&1 | tail -1

# Download Datawhale metrics snapshot
echo ">>> 下载 Datawhale 指标快照"
ASSET_DIR="$SCRIPT_DIR/assets"
mkdir -p "$ASSET_DIR"

METRICS_URL="https://raw.githubusercontent.com/datawhalechina/every-embodied/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/assets/metrics_snapshot.json"

if [ ! -f "$ASSET_DIR/metrics_snapshot.json" ]; then
    curl -sL --max-time 30 "$METRICS_URL" -o "$ASSET_DIR/metrics_snapshot.json"
    echo "  ✅ metrics_snapshot.json 下载完成"
else
    echo "  ✅ metrics_snapshot.json 已存在"
fi

echo ""
echo "=== 环境准备完成 ==="
echo ""
echo "运行脚本:"
echo "  python3 dw_reference/01_check_env.py            # 设备检查"
echo "  python3 dw_reference/02_physical_success.py     # 物理成功评估"
echo "  python3 dw_reference/03_act_dagger.py           # ACT 诊断数据"
echo "  python3 dw_reference/04_smolvla_weighted.py     # SmolVLA 红蓝杯"
echo ""
echo "实操 (需 GPU + 数据):"
echo "  python3 dw_reference/08_train_act.py            # ACT 训练"
echo "  python3 dw_reference/11_eval_closed_loop.py     # 闭环评估"
