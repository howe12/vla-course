#!/usr/bin/env python3
"""DM0.5 HTTP 推理客户端

替代 SmolVLA 的 gRPC stub，通过 HTTP POST /v1/infer 调用 DM0.5 推理服务。
输出单位: 度 (degree)，与 SmolVLA brain_grpc_server_real.py 一致。

用法:
    from brain_http_client import DM05HTTPClient
    client = DM05HTTPClient("http://127.0.0.1:7891")
    actions, latency_ms = client.predict(state_deg, images_jpeg, instruction)
"""
import base64
import io
import time
import logging
import requests
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger("dm05_http")

DEFAULT_ADDR = "http://127.0.0.1:7891"
DEFAULT_INSTRUCTION = "grab two objects into the middle box"
CHUNK_SIZE = 50
ACTION_DIM = 14


class DM05HTTPClient:
    """DM0.5 HTTP 推理客户端"""

    def __init__(self, addr: str = DEFAULT_ADDR, timeout: int = 30):
        self.addr = addr.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        logger.info(f"DM05HTTPClient 初始化: {self.addr}")

    def ping(self) -> bool:
        """测试连接是否可用"""
        try:
            # DM0.5 没有专用 ping 端点，用 GET / 测试
            resp = self.session.get(self.addr, timeout=5)
            return resp.status_code < 500
        except Exception as e:
            logger.warning(f"Ping 失败: {e}")
            return False

    def predict(
        self,
        state_deg: list[float],
        images_jpeg: dict[str, bytes],
        instruction: str = DEFAULT_INSTRUCTION,
        robot_type: str = "Aloha",
    ) -> tuple[np.ndarray, float]:
        """单次推理

        Args:
            state_deg: 14维关节位置，单位: 度 (degree)
            images_jpeg: {"front": jpeg_bytes, "left": jpeg_bytes, "right": jpeg_bytes}
            instruction: 任务指令
            robot_type: 机器人类型 (Aloha / DOS W1)

        Returns:
            actions: np.ndarray shape (50, 14)，单位: 度
            latency_ms: 云端推理延迟 (ms)
        """
        # 编码图片为 base64（front 相机裁剪下半部分以匹配训练视角）
        images_b64 = {}
        cam_order = [("front", "1"), ("left", "2"), ("right", "3")]
        # 生成一个灰色占位 JPEG（避免空字符串被服务端拒绝）
        placeholder_jpeg = None
        for cam_name, img_key in cam_order:
            if cam_name in images_jpeg:
                jpeg_data = images_jpeg[cam_name]
                # Front 相机：裁剪下半部分并 resize 回 480x640
                if cam_name == "front" and HAS_CV2:
                    arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        h = frame.shape[0]
                        cropped = frame[h // 2:, :]  # 下半部分
                        resized = cv2.resize(cropped, (640, 480))  # resize 回模型期望尺寸
                        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ok:
                            jpeg_data = buf.tobytes()
                images_b64[img_key] = base64.b64encode(jpeg_data).decode()
            else:
                logger.warning(f"相机 {cam_name} 缺失，使用灰色占位图")
                if placeholder_jpeg is None:
                    import io
                    try:
                        from PIL import Image
                        buf = io.BytesIO()
                        Image.new("RGB", (640, 480), (128, 128, 128)).save(buf, "JPEG")
                        placeholder_jpeg = buf.getvalue()
                    except ImportError:
                        # 最小有效 JPEG: 1x1 gray pixel
                        placeholder_jpeg = bytes([
                            0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
                            0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
                            0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
                            0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
                            0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
                            0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
                            0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
                            0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
                            0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
                            0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
                            0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
                            0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
                            0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,
                            0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,
                            0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,
                            0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,
                            0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,
                            0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,
                            0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
                            0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,
                            0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
                            0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,
                            0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,
                            0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,
                            0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,
                            0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,
                            0xF5,0xF6,0xF7,0xF8,0xF9,0xFA,0xFF,0xDA,0x00,0x08,0x01,0x01,
                            0x00,0x00,0x3F,0x00,0x7B,0x94,0x11,0x00,0x00,0x00,0x00,0x00,
                            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0xD9])
                images_b64[img_key] = base64.b64encode(placeholder_jpeg).decode()

        payload = {
            "observation": {
                "prompt": instruction,
                "state": state_deg[:ACTION_DIM],
                "images": images_b64,
                "robot_type": robot_type,
                "control_mode": "joint",
                "speed": "0.5",
            }
        }

        t0 = time.time()
        resp = self.session.post(
            f"{self.addr}/v1/infer",
            json=payload,
            timeout=self.timeout,
        )
        total_ms = (time.time() - t0) * 1000

        if resp.status_code != 200:
            raise RuntimeError(f"DM05 推理失败: HTTP {resp.status_code}: {resp.text[:200]}")

        result = resp.json()
        actions = np.array(result["actions"], dtype=np.float32)  # (50, 14)
        server_latency = result.get("metadata", {}).get("latency_ms", 0)

        if actions.shape != (CHUNK_SIZE, ACTION_DIM):
            logger.warning(
                f"Action shape 异常: {actions.shape}, 期望 ({CHUNK_SIZE}, {ACTION_DIM})"
            )

        logger.info(
            f"推理完成: 总 {total_ms:.0f}ms (云端 {server_latency:.0f}ms), "
            f"shape={actions.shape}"
        )
        return actions, server_latency
