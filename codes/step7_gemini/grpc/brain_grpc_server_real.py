#!/usr/bin/env python3
"""云端大脑 gRPC server — 真实 SmolVLA 推理版

实现:
  - Ping: 回显时间戳，测 RTT
  - Predict: 真实 SmolVLA 推理（3路图像 + 14维 state + 语言指令）
  - PredictStream: 双向流推理

启动: python3 brain_grpc_server_real.py --port 50051 [--model /path/to/pretrained_model]
"""
import argparse
import os
import time
from concurrent import futures

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch
import grpc

import embodied_brain_pb2 as pb2
import embodied_brain_pb2_grpc as pb2_grpc

# ── 模型常量（对齐训练 checkpoint）──
DEFAULT_MODEL = "/root/gpufree-data/robotics-vla/checkpoints/smolvla_grasp_two_obj/checkpoints/002000/pretrained_model"
VLM_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
CAMERA_KEYS = ["observation.images.front", "observation.images.left", "observation.images.right"]
STATE_DIM = 14
ACTION_DIM = 14
CHUNK_SIZE = 5
TOKEN_MAX_LEN = 48
DEFAULT_TASK = "grab two objects into the middle box"


class SmolVLAServicer(pb2_grpc.BrainServiceServicer):
    def __init__(self, model_path: str, device: str = "cuda"):
        t0 = time.time()
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from transformers import AutoTokenizer

        # 加载策略
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(model_path)
        self.policy.to(device).eval()

        # 加载语言 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(VLM_NAME)
        print(f"[brain] model+tokenizer 加载完成: {time.time()-t0:.1f}s")

        # 加载反归一化参数（action mean/std）→ 0.1° 真实电机位置
        action_stats = self._load_unnormalizer_stats(model_path)
        self.action_mean = action_stats["mean"].to(device)
        self.action_std = action_stats["std"].to(device)
        print(f"[brain] 反归一化 stats 加载: mean[0:3]={self.action_mean[:3].tolist()}")

    def _load_unnormalizer_stats(self, model_path: str):
        """从 checkpoint postprocessor safetensors 读取 action mean/std"""
        import json
        from pathlib import Path
        import safetensors.torch

        post = json.loads(Path(f"{model_path}/policy_postprocessor.json").read_text())
        # 找 unnormalizer step 的 state_file
        state_file = None
        for step in post["steps"]:
            if step["registry_name"] == "unnormalizer_processor":
                state_file = step.get("state_file")
                break
        if state_file is None:
            raise RuntimeError("postprocessor 中找不到 unnormalizer_processor")

        state = safetensors.torch.load_file(f"{model_path}/{state_file}")
        return {
            "mean": state["action.mean"].float(),
            "std": state["action.std"].float(),
        }

    def _tokenize(self, instruction: str):
        """语言指令 → tokens + attention_mask（bool）"""
        enc = self.tokenizer(
            instruction,
            return_tensors="pt",
            max_length=TOKEN_MAX_LEN,
            truncation=True,
            padding="max_length",
        )
        return (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device).bool(),
        )

    def _jpeg_to_img(self, jpeg_bytes: bytes, width: int, height: int):
        """JPEG 字节 → (3,H,W) float /255 tensor (cuda)"""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        # HWC → CHW, float /255, 加 batch 维
        img = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return img.to(self.device)

    def _obs_to_batch(self, obs: pb2.Observation) -> dict:
        """Observation → 模型 batch dict"""
        # 图像：按 camera_name 映射到 front/left/right
        img_map = {}
        for img in obs.images:
            name = img.camera_name  # "front"/"left"/"right" 或索引
            if name in img_map:
                continue
            img_map[name] = self._jpeg_to_img(img.jpeg_data, img.width, img.height)

        # 如果相机名不匹配，按顺序填充
        for i, ck in enumerate(CAMERA_KEYS):
            if ck not in img_map and i < len(obs.images):
                img = obs.images[i]
                img_map[ck] = self._jpeg_to_img(img.jpeg_data, img.width, img.height)

        batch = {ck: img_map[ck] for ck in CAMERA_KEYS if ck in img_map}
        # 缺相机的补零，避免形状不匹配
        for ck in CAMERA_KEYS:
            if ck not in batch:
                batch[ck] = torch.zeros(1, 3, 480, 640).to(self.device)

        # state
        state = list(obs.state)
        if len(state) < STATE_DIM:
            state += [0.0] * (STATE_DIM - len(state))
        batch["observation.state"] = torch.tensor([state[:STATE_DIM]], dtype=torch.float32).to(self.device)

        # language
        instruction = obs.instruction if obs.instruction else DEFAULT_TASK
        tokens, mask = self._tokenize(instruction)
        batch["observation.language.tokens"] = tokens
        batch["observation.language.attention_mask"] = mask
        return batch

    def _predict(self, obs: pb2.Observation) -> pb2.ActionChunk:
        """单次推理"""
        t0 = time.time()
        batch = self._obs_to_batch(obs)
        with torch.no_grad():
            raw_actions = self.policy.predict_action_chunk(batch)
        inference_ms = int((time.time() - t0) * 1000)

        # 反归一化: 归一化空间 → 真实电机位置 (0.1° 单位)
        # raw: (1, chunk_size, action_dim); real = raw * std + mean
        real_actions = raw_actions * self.action_std + self.action_mean
        chunk = real_actions[0].float().cpu().numpy()
        # 展平: actions[i*action_dim + j]
        flat = chunk.flatten().tolist()
        return pb2.ActionChunk(
            actions=flat,
            chunk_size=CHUNK_SIZE,
            action_dim=ACTION_DIM,
            inference_ms=inference_ms,
            server_timestamp_ms=int(time.time() * 1000),
        )

    def Ping(self, request, context):
        return pb2.PingResponse(
            client_timestamp_ms=request.timestamp_ms,
            server_timestamp_ms=int(time.time() * 1000),
            payload=request.payload,
        )

    def Predict(self, request, context):
        return self._predict(request)

    def PredictStream(self, request_iterator, context):
        for obs in request_iterator:
            yield self._predict(obs)


def serve(port, model_path):
    servicer = SmolVLAServicer(model_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_BrainServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    print(f"[brain] 真实 SmolVLA gRPC server 监听 0.0.0.0:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    serve(args.port, args.model)