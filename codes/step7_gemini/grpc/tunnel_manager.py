#!/usr/bin/env python3
"""SSH 隧道管理器（带健康检查 + 指数退避 + 状态文件）

替代裸 systemd ssh，提供：
  1. 自动重连（指数退避：5s→10s→20s→...→max 300s）
  2. 健康检查（TCP 探测 localhost:50051）
  3. 状态文件（/tmp/grpc_tunnel_status.json）供上层应用读取
  4. 连续失败计数 + 云端不可用标记

用法:
    python3 tunnel_manager.py
    # 或由 systemd 管理: ExecStart=python3 /home/nxrobo/grpc_embodied/tunnel_manager.py
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time

# 配置
SSH_HOST = "120.209.70.195"
SSH_PORT = 30334
SSH_USER = "root"
LOCAL_PORT = 50051
REMOTE_PORT = 50051
STATUS_FILE = "/tmp/grpc_tunnel_status.json"

# 重试策略
INITIAL_BACKOFF = 5       # 初始重试间隔（秒）
MAX_BACKOFF = 300         # 最大重试间隔（秒）
BACKOFF_MULTIPLIER = 2    # 退避倍数
HEALTH_CHECK_INTERVAL = 10  # 健康检查间隔（秒）
MAX_CONSECUTIVE_FAILURES = 6  # 连续失败多少次标记为"不可用"（约 5+10+20+40+80+160=315s）


class TunnelManager:
    def __init__(self):
        self.ssh_process = None
        self.backoff = INITIAL_BACKOFF
        self.consecutive_failures = 0
        self.total_reconnects = 0
        self.last_connected_time = None
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\n[tunnel] 收到信号 {signum}，正在退出...")
        self.running = False
        self._stop_ssh()

    def _write_status(self, connected: bool, message: str = ""):
        """写状态文件，供上层应用读取"""
        status = {
            "connected": connected,
            "consecutive_failures": self.consecutive_failures,
            "total_reconnects": self.total_reconnects,
            "last_connected_time": self.last_connected_time,
            "last_check_time": time.time(),
            "message": message,
            "cloud_available": self.consecutive_failures < MAX_CONSECUTIVE_FAILURES,
        }
        try:
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass

    def _start_ssh(self) -> bool:
        """启动 SSH 隧道进程"""
        cmd = [
            "ssh", "-N",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ConnectTimeout=15",
            "-L", f"{LOCAL_PORT}:localhost:{REMOTE_PORT}",
            "-p", str(SSH_PORT),
            f"{SSH_USER}@{SSH_HOST}",
        ]
        try:
            self.ssh_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # 等待一小段时间确认进程启动
            time.sleep(3)
            if self.ssh_process.poll() is not None:
                stderr = self.ssh_process.stderr.read().decode() if self.ssh_process.stderr else ""
                print(f"[tunnel] SSH 进程立即退出: {stderr[:200]}")
                return False
            return True
        except Exception as e:
            print(f"[tunnel] 启动 SSH 失败: {e}")
            return False

    def _stop_ssh(self):
        """停止 SSH 隧道进程"""
        if self.ssh_process and self.ssh_process.poll() is None:
            self.ssh_process.terminate()
            try:
                self.ssh_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ssh_process.kill()
        self.ssh_process = None

    def _health_check(self) -> bool:
        """TCP 探测本地端口，确认隧道连通"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(("127.0.0.1", LOCAL_PORT))
            sock.close()
            return result == 0
        except Exception:
            return False

    def run(self):
        """主循环"""
        print(f"[tunnel] 启动，目标: {SSH_USER}@{SSH_HOST}:{SSH_PORT} → localhost:{LOCAL_PORT}")
        print(f"[tunnel] 状态文件: {STATUS_FILE}")

        while self.running:
            # 启动隧道
            print(f"[tunnel] 尝试连接（第 {self.total_reconnects + 1} 次）...")
            if self._start_ssh():
                # 健康检查确认
                if self._health_check():
                    print(f"[tunnel] ✅ 隧道已建立 (localhost:{LOCAL_PORT} → 云端:{REMOTE_PORT})")
                    self.consecutive_failures = 0
                    self.backoff = INITIAL_BACKOFF
                    self.last_connected_time = time.time()
                    self._write_status(True, "connected")

                    # 持续健康检查
                    self._monitor_loop()
                else:
                    print("[tunnel] SSH 进程启动但端口未监听，重试...")
                    self._stop_ssh()
                    self.consecutive_failures += 1
                    self._write_status(False, "port not listening")
            else:
                self.consecutive_failures += 1
                self._write_status(False, "ssh start failed")

            if not self.running:
                break

            # 指数退避
            self.total_reconnects += 1
            cloud_available = self.consecutive_failures < MAX_CONSECUTIVE_FAILURES
            if not cloud_available:
                print(f"[tunnel] ⚠️ 云端连续 {self.consecutive_failures} 次不可用，"
                      f"标记为离线，{self.backoff}s 后重试")
            else:
                print(f"[tunnel] 连接失败（第 {self.consecutive_failures} 次），"
                      f"{self.backoff}s 后重试")

            self._write_status(False, f"retrying in {self.backoff}s")
            time.sleep(self.backoff)
            self.backoff = min(self.backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

        self._stop_ssh()
        self._write_status(False, "manager stopped")
        print("[tunnel] 已退出")

    def _monitor_loop(self):
        """隧道已建立后的持续监控"""
        while self.running:
            time.sleep(HEALTH_CHECK_INTERVAL)
            if not self.running:
                break

            # 检查 SSH 进程是否还活着
            if self.ssh_process and self.ssh_process.poll() is not None:
                print("[tunnel] SSH 进程已退出")
                break

            # TCP 健康检查
            if not self._health_check():
                print("[tunnel] 健康检查失败（端口不可达）")
                self.consecutive_failures += 1
                self._write_status(False, "health check failed")
                if self.consecutive_failures >= 3:
                    print("[tunnel] 连续 3 次健康检查失败，重建隧道")
                    self._stop_ssh()
                    break
            else:
                if self.consecutive_failures > 0:
                    print("[tunnel] 健康检查恢复")
                self.consecutive_failures = 0
                self._write_status(True, "healthy")


if __name__ == "__main__":
    manager = TunnelManager()
    manager.run()
