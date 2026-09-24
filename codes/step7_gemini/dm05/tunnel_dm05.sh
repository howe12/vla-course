#!/bin/bash
# DM0.5 SSH 隧道: NUC localhost:7891 → 云端推理服务 :7891
#
# 用法:
#   export DM05_REMOTE_HOST=your.gpu.host
#   export DM05_REMOTE_PORT=22          # SSH 端口（容器常见为映射出的高位端口）
#   ./tunnel_dm05.sh                    # 前台运行
#   ./tunnel_dm05.sh &                  # 后台运行
#
# 验证:
#   curl http://127.0.0.1:7891/
#
# 注意：不要用「本地机 → 云端」再「本地机 → NUC」的两跳转发去给 NUC 供推理服务。
# 实测两跳链路上推理延迟尖峰可达 14925ms，超过执行器 10s 超时阈值 → 触发熔断，
# 60% 的回合被整轮跳过。改成 NUC 直连云端（NUC 自身公钥加入云端
# authorized_keys）后：平均延迟 1439→836ms，最大 14925→1306ms，熔断 0 次。

set -euo pipefail

REMOTE_HOST="${DM05_REMOTE_HOST:?请设置 DM05_REMOTE_HOST 为推理服务所在主机}"
REMOTE_PORT="${DM05_REMOTE_PORT:-22}"
REMOTE_USER="${DM05_REMOTE_USER:-root}"
LOCAL_PORT="${DM05_LOCAL_PORT:-7891}"
REMOTE_SERVICE_PORT="${DM05_REMOTE_SERVICE_PORT:-7891}"

echo "=== DM0.5 SSH 隧道 ==="
echo "  本地: localhost:${LOCAL_PORT}"
echo "  远程: ${REMOTE_HOST}:${REMOTE_SERVICE_PORT} (via SSH port ${REMOTE_PORT})"
echo ""
echo "按 Ctrl+C 停止隧道"
echo ""

# ServerAliveInterval 让隧道在链路静默时保持存活；推理请求间隔较长，
# 不加会被中间设备回收连接。
exec ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_SERVICE_PORT}" \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -p "${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}"
