#!/usr/bin/env python3
"""Orin 小脑 gRPC client（延迟测试版）

测试:
  1. Ping RTT（多次往返延迟统计）
  2. Predict 单次推理延迟
  3. 不同 payload 的传输延迟
"""
import time
import statistics

import grpc
import embodied_brain_pb2 as pb2
import embodied_brain_pb2_grpc as pb2_grpc

SERVER = "localhost:50051"  # 通过 SSH 隧道映射


def test_ping_rtt(stub, n=20):
    """测试 Ping 往返延迟"""
    rtts = []
    for _ in range(n):
        t0 = time.time()
        resp = stub.Ping(pb2.PingRequest(timestamp_ms=int(t0 * 1000), payload="x" * 100))
        rtt = (time.time() - t0) * 1000
        rtts.append(rtt)
    return rtts


def test_predict_latency(stub, n=10):
    """测试 Predict（含图像）往返延迟"""
    # 模拟 2 路 256x256 图像（JPEG 压缩后约 10-30KB）
    fake_jpeg = b"\xff\xd8" + b"\x00" * 20000 + b"\xff\xd9"  # 模拟 ~20KB JPEG
    latencies = []
    for _ in range(n):
        obs = pb2.Observation(
            images=[
                pb2.Image(camera_name="image", jpeg_data=fake_jpeg, width=256, height=256),
                pb2.Image(camera_name="image2", jpeg_data=fake_jpeg, width=256, height=256),
            ],
            state=[0.1] * 8,
            instruction="pick up the object into the box",
            timestamp_ms=int(time.time() * 1000),
        )
        t0 = time.time()
        resp = stub.Predict(obs)
        latency = (time.time() - t0) * 1000
        latencies.append(latency)
    return latencies, resp


def stats(name, values):
    print(f"  {name}:")
    print(f"    次数: {len(values)}")
    print(f"    平均: {statistics.mean(values):.2f} ms")
    print(f"    最小: {min(values):.2f} ms")
    print(f"    最大: {max(values):.2f} ms")
    print(f"    中位: {statistics.median(values):.2f} ms")
    if len(values) > 1:
        print(f"    标准差: {statistics.stdev(values):.2f} ms")


def main():
    print(f"=== 连接 gRPC server: {SERVER} ===")
    channel = grpc.insecure_channel(SERVER)
    grpc.channel_ready_future(channel).result(timeout=10)
    stub = pb2_grpc.BrainServiceStub(channel)
    print("连接成功\n")

    # 预热
    print("=== 预热（5 次 Ping）===")
    for _ in range(5):
        stub.Ping(pb2.PingRequest(timestamp_ms=int(time.time() * 1000)))
    print("预热完成\n")

    # 测试 1: Ping RTT
    print("=== 测试 1: Ping RTT（20 次）===")
    rtts = test_ping_rtt(stub, 20)
    stats("Ping RTT", rtts)
    print()

    # 测试 2: Predict 延迟（含 2 路图像）
    print("=== 测试 2: Predict 延迟（2 路 256x256 图像，10 次）===")
    latencies, resp = test_predict_latency(stub, 10)
    stats("Predict 往返延迟", latencies)
    print(f"  返回动作: chunk_size={resp.chunk_size}, action_dim={resp.action_dim}")
    print(f"  云端推理耗时: {resp.inference_ms} ms")
    print(f"  动作前 7 维: {list(resp.actions[:7])}")
    print()

    print("=== 测试完成 ===")
    channel.close()


if __name__ == "__main__":
    main()
