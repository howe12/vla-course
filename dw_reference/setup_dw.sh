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

# Install deps
echo ">>> 安装依赖"
pip install -q jupyter nbclient pyyaml 2>&1 | tail -1
pip install -q "lerobot[training]" 2>&1 | tail -1

ASSET_DIR="$SCRIPT_DIR/assets"
mkdir -p "$ASSET_DIR"
BASE_URL="https://raw.githubusercontent.com/datawhalechina/every-embodied/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/assets"

echo ">>> 下载 Datawhale 示例数据"
for f in metrics_snapshot.json collection_dataset_snapshot.json; do
    if [ ! -f "$ASSET_DIR/$f" ]; then
        curl -sL --max-time 30 "$BASE_URL/$f" -o "$ASSET_DIR/$f"
        echo "  ✅ $f 下载完成"
    else
        echo "  ✅ $f 已存在"
    fi
done

echo ""
echo "=== 环境准备完成 ==="
echo ""
echo "信息展示型 (无需 GPU):"
echo "  python3 dw_reference/01_check_env.py            # 设备检查"
echo "  python3 dw_reference/02_physical_success.py     # 物理成功评估"
echo "  python3 dw_reference/03_act_dagger.py           # ACT 诊断数据"
echo "  python3 dw_reference/04_smolvla_weighted.py     # SmolVLA 红蓝杯"
echo "  python3 dw_reference/07_audit_data.py --demo    # 数据集审计 (演示)"
echo ""
echo "实操型 (需 GPU):"
echo "  python3 dw_reference/07_audit_data.py --dataset /path/to/ds  # 数据集审计"
echo "  python3 dw_reference/08_train_act.py --dataset /path/to/ds --smoke  # ACT smoke"
echo "  RUN_REAL_TRAIN=1 python3 dw_reference/08_train_act.py --dataset /path/to/ds --train  # ACT 训练"
