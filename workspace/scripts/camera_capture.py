#!/usr/bin/env python3
"""Orin 相机采集模块

采集 2 路相机（video0/video2），缩放到 256x256（对齐 smolvla_libero），
JPEG 编码后封装为 gRPC Image 消息。

用法:
    from camera_capture import CameraCapture
    cap = CameraCapture()
    images = cap.capture()  # 返回 [pb2.Image, pb2.Image]
    cap.release()
"""
import cv2
import numpy as np

# 相机映射（video0=主相机, video2=副相机; video1/video3 是元数据节点）
CAMERA_MAP = {
    "top": "/dev/v4l/by-id/usb-Generic_USB_Camera3_200901010001-video-index0",   # 顶部视角（外接 USB Camera3）
    "front": "/dev/v4l/by-id/usb-170428-_Integrated_Webcam_HD-video-index0",     # 前部视角（集成 Webcam）
}
TARGET_SIZE = (256, 256)  # smolvla_libero 输入尺寸
JPEG_QUALITY = 85


class CameraCapture:
    def __init__(self, camera_map=None, target_size=TARGET_SIZE, jpeg_quality=JPEG_QUALITY):
        self.camera_map = camera_map or CAMERA_MAP
        self.target_size = target_size
        self.jpeg_quality = jpeg_quality
        self.caps = {}
        self._open_cameras()

    def _open_cameras(self):
        for name, path in self.camera_map.items():
            # 稳定绑定：用 /dev/v4l/by-id 符号链接（按 USB 序列号，拔插/重启不换）
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开相机 {name} ({path})")
            # 设置请求分辨率（相机可能不支持，会回退到最近值）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.caps[name] = cap
            print(f"[camera] 打开 {name} ({path})")

    def capture(self):
        """采集所有相机，返回 {name: jpeg_bytes} 字典"""
        result = {}
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"相机 {name} 读帧失败")
            # BGR → 缩放到目标尺寸（保持宽高比，padding 到正方形）
            frame = self._resize_with_pad(frame, self.target_size)
            # JPEG 编码
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                raise RuntimeError(f"相机 {name} JPEG 编码失败")
            result[name] = {
                "jpeg_data": buf.tobytes(),
                "width": self.target_size[0],
                "height": self.target_size[1],
            }
        return result

    def capture_as_pb2(self, pb2):
        """采集并封装为 gRPC Image 消息列表"""
        data = self.capture()
        images = []
        for name, info in data.items():
            images.append(pb2.Image(
                camera_name=name,
                jpeg_data=info["jpeg_data"],
                width=info["width"],
                height=info["height"],
            ))
        return images

    @staticmethod
    def _resize_with_pad(frame, target_size):
        """保持宽高比缩放，padding 到目标尺寸（对齐 lerobot 的 resize_with_pad）"""
        tw, th = target_size
        h, w = frame.shape[:2]
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 创建黑色画布，居中放置
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        y_off, x_off = (th - new_h) // 2, (tw - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def release(self):
        for cap in self.caps.values():
            cap.release()
        print("[camera] 已释放所有相机")


if __name__ == "__main__":
    # 独立测试：采集并保存预览图
    cap = CameraCapture()
    try:
        for i in range(3):
            data = cap.capture()
            for name, info in data.items():
                print(f"{name}: {len(info['jpeg_data'])} bytes, {info['width']}x{info['height']}")
            import time
            time.sleep(0.5)
    finally:
        cap.release()
