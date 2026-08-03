#!/usr/bin/env python3
"""双相机可视化验证脚本

启动两个相机（video0/video2），并排显示 + 标注相机名，
按 [S] 保存快照，按 [Q] 退出。用于确认相机绑定是否正确。

用法:
    python3 camera_preview.py            # 正常模式（窗口显示）
    python3 camera_preview.py --snap 3   # 只拍 3 张快照保存，不显示窗口

依赖: OpenCV (lerobot_061 环境已装 cv2 4.13)
"""
import argparse
import os
import time

import cv2
import numpy as np

# 相机映射（用户确认的物理含义，2026-08-03）
# top = 全局俯视（外接 USB Camera3, video0），front = 平视视角（集成 Webcam, video2）
# left/right = 双臂相机（当前未接，LEO-Gemini 共 4 相机）
CAMERAS = [
    {"name": "top", "path": "/dev/v4l/by-id/usb-Generic_USB_Camera3_200901010001-video-index0", "desc": "全局俯视 (USB Camera3)"},
    {"name": "front", "path": "/dev/v4l/by-id/usb-170428-_Integrated_Webcam_HD-video-index0", "desc": "平视视角 (Integrated Webcam)"},
]

SNAP_DIR = os.path.expanduser("~/camera_snapshots")


def open_camera(path):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"  ❌ {path}: 无法打开")
        return None
    # 尝试设置更高分辨率（若相机支持）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def grab_frame(cap, name, desc):
    """采集一帧并标注相机名"""
    ret, frame = cap.read()
    if not ret:
        return None
    # 标注相机名和描述
    label = f"{name} | {desc}"
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def make_stacked(cam_frames):
    """将多路帧并排拼接（高度对齐）"""
    if not cam_frames:
        return None
    h = max(f.shape[0] for f in cam_frames)
    padded = []
    for f in cam_frames:
        if f.shape[0] < h:
            # 补齐高度（黑色）
            pad = np.zeros((h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            f = np.vstack([f, pad])
        padded.append(f)
    return np.hstack(padded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=int, default=0,
                    help="只拍摄 N 张快照并保存（不显示窗口），0=实时窗口模式")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="快照间隔秒数（snap 模式）")
    args = ap.parse_args()

    # 打开相机
    caps = {}
    for cam in CAMERAS:
        print(f"打开 {cam['name']} ({cam['desc']}) ...")
        cap = open_camera(cam["path"])
        if cap:
            caps[cam["name"]] = {"cap": cap, "info": cam}
        else:
            print(f"  ⚠️ {cam['name']} 打开失败，跳过")

    if not caps:
        print("❌ 两个相机都打不开，请检查设备连接")
        return

    print(f"\n✅ 已打开 {len(caps)} 个相机")
    for name, c in caps.items():
        w = c["cap"].get(cv2.CAP_PROP_FRAME_WIDTH)
        h = c["cap"].get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"  {name}: {int(w)}x{int(h)}")

    os.makedirs(SNAP_DIR, exist_ok=True)

    if args.snap > 0:
        # 快照模式：连拍 N 张保存
        print(f"\n📸 快照模式：拍摄 {args.snap} 张，保存到 {SNAP_DIR}")
        for i in range(args.snap):
            frames = [grab_frame(c["cap"], name, c["info"]["desc"])
                      for name, c in caps.items()]
            frames = [f for f in frames if f is not None]
            if frames:
                stacked = make_stacked(frames)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SNAP_DIR, f"camera_{ts}_{i}.jpg")
                cv2.imwrite(path, stacked)
                print(f"  ✅ {path} ({stacked.shape[1]}x{stacked.shape[0]})")
            else:
                print(f"  ⚠️ 第 {i+1} 张采集失败")
            time.sleep(args.interval)
        print(f"\n完成！查看: {SNAP_DIR}")
        for f in sorted(os.listdir(SNAP_DIR))[-args.snap:]:
            print(f"  {os.path.join(SNAP_DIR, f)}")
        print("可用 scp 拉取或用 VNC 打开图片查看")
    else:
        # 实时窗口模式（通过 VNC 可看到）
        print("\n🖥️  实时预览（通过 VNC 查看），按 [S] 保存快照，按 [Q] 退出")
        while True:
            frames = [grab_frame(c["cap"], name, c["info"]["desc"])
                      for name, c in caps.items()]
            frames = [f for f in frames if f is not None]
            if frames:
                stacked = make_stacked(frames)
                cv2.imshow("Dual Camera (image | image2)", stacked)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("退出")
                break
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SNAP_DIR, f"camera_{ts}.jpg")
                cv2.imwrite(path, stacked)
                print(f"  📸 已保存: {path}")

    for c in caps.values():
        c["cap"].release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless 环境（无 GTK）下忽略


if __name__ == "__main__":
    main()
