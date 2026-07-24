#!/usr/bin/env python3
"""
云端大脑 API Stub (L40 容器)

接收 Gemini 小脑发来的观测 → 调用 VLA 模型推理 → 返回动作。

启动: python3 brain_server.py --port 8001
测试: curl -X POST localhost:8001/predict -H 'Content-Type: application/json' -d '{"images":{},"joint_state":[],"instruction":"pick apple"}'
"""

import argparse, json
from http.server import HTTPServer, BaseHTTPRequestHandler

class BrainHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404); return
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        # TODO: 替换为真实的 VLA 推理
        action = [0.01, 0.0, -0.02, 0.0, 0.0, 0.0, 1.0]  # 示例：向下移动 2cm，开夹爪
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"action": action}).encode())

def main(port):
    srv = HTTPServer(("0.0.0.0", port), BrainHandler)
    print(f"Brain API listening on :{port}")
    srv.serve_forever()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    main(ap.parse_args().port)
