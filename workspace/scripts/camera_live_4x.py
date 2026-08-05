#!/usr/bin/env python3
"""相机实时采集可视化（模拟采集效果）

稳定组合（USB2 带宽限制，bus1 最多 2 个 Microdia）：
  --mode 3a : top + right + left（推荐，实测 20/20 稳定）
  --mode 3b : top + front + left（数据集常用组合）
  --mode 4  : top + front + left + right（第 3 个 Microdia 会失败）

用法（在 VNC 桌面终端）:
    conda activate lerobot_061
    python3 camera_live_4x.py --mode 3a
    python3 camera_live_4x.py --mode 3b
"""
import argparse
import time
import cv2
import numpy as np

# 相机定义（by-path 稳定绑定，用户确认的空间位置）
CAMERAS = {
    "top":   {"pos": "左上·全局俯视", "path": "/dev/v4l/by-path/platform-3610000.usb-usb-0:1.1:1.0-video-index0", "fourcc": None, "fps": 30},
    "front": {"pos": "右上·平视视角", "path": "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0", "fourcc": "MJPG", "fps": 25},
    "left":  {"pos": "左下·左臂",     "path": "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2.2:1.0-video-index0", "fourcc": "MJPG", "fps": 25},
    "right": {"pos": "右下·右臂",     "path": "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0", "fourcc": "MJPG", "fps": 25},
}
WIDTH, HEIGHT = 640, 480
CELL_W, CELL_H = 320, 240
SNAP_DIR = "/home/nxrobo/camera_snapshots"

# 组合定义
MODES = {
    "3a": ["top", "right", "left"],
    "3b": ["top", "front", "left"],
    "4":  ["top", "front", "left", "right"],
}


def open_cameras(names):
    caps = {}
    for name in names:
        cam = CAMERAS[name]
        cap = cv2.VideoCapture(cam["path"], cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"  [FAIL] {name}: 无法打开")
            caps[name] = None
            continue
        if cam["fourcc"]:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, cam["fps"])
        caps[name] = cap
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fmt = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        print(f"  [OK] {name}: {fmt} {WIDTH}x{HEIGHT} @ {cam['pos']}")
        time.sleep(0.5)
    return caps


def make_placeholder(name, msg):
    cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    cv2.putText(cell, name, (10, CELL_H // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(cell, msg, (10, CELL_H // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="3a", choices=list(MODES.keys()))
    args = parser.parse_args()
    names = MODES[args.mode]

    print(f"=== 打开 {len(names)} 个相机（mode={args.mode}）===")
    caps = open_cameras(names)
    ok = sum(1 for c in caps.values() if c is not None)
    print(f"成功: {ok}/{len(names)}\n")
    if ok == 0:
        return

    print("实时预览中... [S]=保存快照  [Q]=退出")
    fps_count = {n: 0 for n in caps}
    fps_val = {n: 0.0 for n in caps}
    fps_t = time.time()
    last_frames = {}

    while True:
        cells = []
        for name in names:
            cap = caps.get(name)
            if cap is None:
                cells.append(make_placeholder(name, "N/A"))
                continue
            ret, frame = cap.read()
            if not ret:
                fps_count[name] = 0
                cells.append(make_placeholder(name, "NO FRAME"))
                continue
            last_frames[name] = frame
            fps_count[name] += 1
            cell = cv2.resize(frame, (CELL_W, CELL_H))
            label = f"{name} {fps_val[name]:.0f}fps"
            cv2.putText(cell, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cells.append(cell)

        # 补齐到 4 格
        while len(cells) < 4:
            cells.append(make_placeholder("", ""))
        grid = np.vstack([np.hstack(cells[0:2]), np.hstack(cells[2:4])])
        cv2.imshow("Camera Live Preview", grid)

        now = time.time()
        if now - fps_t >= 1.0:
            for n in caps:
                fps_val[n] = fps_count[n] / (now - fps_t)
                fps_count[n] = 0
            fps_t = now

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            for n, f in last_frames.items():
                cv2.imwrite(f"{SNAP_DIR}/live_{n}_{ts}.jpg", f)
            cv2.imwrite(f"{SNAP_DIR}/live_grid_{ts}.jpg", grid)
            print(f"  快照已保存: {SNAP_DIR}/live_grid_{ts}.jpg")

    for c in caps.values():
        if c is not None:
            c.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
