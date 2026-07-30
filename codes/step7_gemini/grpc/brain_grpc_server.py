#!/usr/bin/env python3
"""云端大脑 gRPC server（echo/延迟测试版）

实现:
  - Ping: 回显时间戳，测 RTT
  - Predict: stub 返回固定动作（TODO: 接真实 VLA 推理）
  - PredictStream: 双向流（回显观测，返回固定动作）

启动: python3 brain_grpc_server.py --port 50051
"""
import argparse
import time
from concurrent import futures

import grpc
import embodied_brain_pb2 as pb2
import embodied_brain_pb2_grpc as pb2_grpc


class BrainServicer(pb2_grpc.BrainServiceServicer):
    def Ping(self, request, context):
        return pb2.PingResponse(
            client_timestamp_ms=request.timestamp_ms,
            server_timestamp_ms=int(time.time() * 1000),
            payload=request.payload,
        )

    def Predict(self, request, context):
        t0 = time.time()
        # TODO: 替换为真实 VLA 推理（加载 smolvla_libero）
        action_dim = 7
        chunk_size = 50
        actions = [0.0] * (chunk_size * action_dim)
        actions[6] = 1.0  # 示例：第一步夹爪张开
        inference_ms = int((time.time() - t0) * 1000)
        return pb2.ActionChunk(
            actions=actions,
            chunk_size=chunk_size,
            action_dim=action_dim,
            inference_ms=inference_ms,
            server_timestamp_ms=int(time.time() * 1000),
        )

    def PredictStream(self, request_iterator, context):
        for obs in request_iterator:
            t0 = time.time()
            action_dim = 7
            chunk_size = 50
            actions = [0.0] * (chunk_size * action_dim)
            actions[6] = 1.0
            inference_ms = int((time.time() - t0) * 1000)
            yield pb2.ActionChunk(
                actions=actions,
                chunk_size=chunk_size,
                action_dim=action_dim,
                inference_ms=inference_ms,
                server_timestamp_ms=int(time.time() * 1000),
            )


def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_BrainServiceServicer_to_server(BrainServicer(), server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    print(f"[brain] gRPC server listening on 0.0.0.0:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50051)
    args = ap.parse_args()
    serve(args.port)
