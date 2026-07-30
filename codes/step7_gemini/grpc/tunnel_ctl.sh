#!/bin/bash
# ============================================================
# gRPC 隧道管理脚本 — 连接云端大脑 (L40)
# 位置: ~/vla_workspace/scripts/tunnel_ctl.sh
# 用法: ./tunnel_ctl.sh {start|stop|restart|status|logs|test}
# ============================================================

SERVICE_NAME="grpc-tunnel"
STATUS_FILE="/tmp/grpc_tunnel_status.json"
LOCAL_PORT=50051
CLOUD_HOST="120.209.70.195"
CLOUD_SSH_PORT=30334

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

show_status() {
    echo "=== gRPC 隧道状态 ==="
    # systemd 服务状态
    if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
        echo -e "服务: ${GREEN}运行中${NC}"
    else
        echo -e "服务: ${RED}未运行${NC}"
    fi

    # 状态文件
    if [ -f "$STATUS_FILE" ]; then
        echo "--- 隧道详情 ---"
        python3 -c "
import json
s = json.load(open('$STATUS_FILE'))
print(f\"  连接状态: {'✅ 已连接' if s['connected'] else '❌ 断开'}\")
print(f\"  云端可用: {'✅ 是' if s['cloud_available'] else '❌ 否'}\")
print(f\"  连续失败: {s['consecutive_failures']} 次\")
print(f\"  重连次数: {s['total_reconnects']} 次\")
print(f\"  消息: {s['message']}\")
" 2>/dev/null
    else
        echo "  状态文件不存在"
    fi

    # 端口监听
    if ss -tln 2>/dev/null | grep -q ":$LOCAL_PORT "; then
        echo -e "  本地端口: ${GREEN}$LOCAL_PORT 监听中${NC}"
    else
        echo -e "  本地端口: ${RED}$LOCAL_PORT 未监听${NC}"
    fi
}

test_connection() {
    echo "=== 测试 gRPC 连通性 ==="
    # 测试本地端口
    if timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$LOCAL_PORT" 2>/dev/null; then
        echo -e "  隧道 (localhost:$LOCAL_PORT): ${GREEN}连通${NC}"
    else
        echo -e "  隧道 (localhost:$LOCAL_PORT): ${RED}不通${NC}"
        return 1
    fi

    # 测试云端 SSH
    if timeout 5 bash -c "exec 3<>/dev/tcp/$CLOUD_HOST/$CLOUD_SSH_PORT" 2>/dev/null; then
        echo -e "  云端 SSH ($CLOUD_HOST:$CLOUD_SSH_PORT): ${GREEN}可达${NC}"
    else
        echo -e "  云端 SSH ($CLOUD_HOST:$CLOUD_SSH_PORT): ${RED}不可达（云平台可能关机）${NC}"
    fi
}

show_logs() {
    echo "=== 隧道日志（最近 20 行）==="
    sudo journalctl -u $SERVICE_NAME --no-pager -n 20 2>/dev/null || echo "无法读取日志"
}

case "$1" in
    start)
        echo "启动 gRPC 隧道..."
        sudo systemctl start $SERVICE_NAME
        sleep 3
        show_status
        ;;
    stop)
        echo "停止 gRPC 隧道..."
        sudo systemctl stop $SERVICE_NAME
        echo -e "${YELLOW}隧道已停止${NC}"
        ;;
    restart)
        echo "重启 gRPC 隧道..."
        sudo systemctl restart $SERVICE_NAME
        sleep 3
        show_status
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    test)
        test_connection
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "  start   - 启动隧道服务（开机已自启，通常不需要手动）"
        echo "  stop    - 停止隧道"
        echo "  restart - 重启隧道"
        echo "  status  - 查看隧道状态"
        echo "  logs    - 查看最近日志"
        echo "  test    - 测试连通性"
        exit 1
        ;;
esac
