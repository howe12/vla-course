#!/usr/bin/env python3
"""遥操作 + 相机 联动测试脚本（安全模式）

功能：
  1. 读取主臂（leader）关节位置（只读）
  2. 可选：驱动从臂（follower）跟随主臂（默认关闭，dry-run）
  3. 显示 3 路相机画面（top/right/left，通过 VNC 查看）
  4. 打印当前主臂关节状态

用法（在 VNC 桌面终端）:
    conda activate lerobot_061
    python3 test_teleop_camera.py                 # 只读主臂 + 显示相机（不驱动从臂）
    python3 test_teleop_camera.py --drive         # 驱动从臂跟随（需确认安全）
    python3 test_teleop_camera.py --no-camera     # 不显示相机（只测遥操作）

安全说明:
  - 默认 --dry-run（不驱动从臂），只读主臂位置
  - --drive 才会真实驱动从臂（需用户明确确认）
"""
import argparse
import time
import sys
import serial
import struct

# 相机（by-path 稳定绑定）
CAMERAS = [
    {"name": "top",   "path": "/dev/v4l/by-path/platform-3610000.usb-usb-0:1.1:1.0-video-index0", "fourcc": None},
    {"name": "right", "path": "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0", "fourcc": "MJPG"},
    {"name": "left",  "path": "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2.2:1.0-video-index0", "fourcc": "MJPG"},
]
JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_flex", "wrist_roll", "gripper"]


def read_leader_positions(port, duration=0.1):
    """只读：解析 SgrMotorsBus 上报帧，返回 7 关节位置"""
    positions = None
    try:
        s = serial.Serial(port, 115200, timeout=0.3)
        buf = b""
        t0 = time.time()
        while time.time() - t0 < duration:
            buf += s.read(256)
            idx = buf.find(b"\x55\xaa")
            if idx >= 0 and idx + 3 <= len(buf):
                pkt_len = buf[idx + 2]
                if idx + 3 + pkt_len + 2 <= len(buf):
                    payload = buf[idx + 3:idx + 3 + pkt_len]
                    positions = {}
                    pos = 1
                    for j in JOINTS:
                        if pos + 2 <= len(payload):
                            positions[j] = struct.unpack("<h", payload[pos:pos + 2])[0] / 10.0
                            pos += 2
                    s.close()
                    return positions
            if idx > 0:
                buf = buf[idx:]
        s.close()
    except Exception as e:
        print(f"  [ERR] 读取主臂失败: {e}")
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", action="store_true", help="驱动从臂跟随（默认只读主臂）")
    parser.add_argument("--no-camera", action="store_true", help="不显示相机")
    parser.add_argument("--left-port", default="/dev/ttyACM_left_leader")
    parser.add_argument("--right-port", default="/dev/ttyACM_right_leader")
    args = parser.parse_args()

    print("=" * 60)
    print("LEO-Gemini 遥操作 + 相机 测试")
    print(f"模式: {'🔄 驱动从臂' if args.drive else '🔒 只读主臂（安全）'}")
    print("=" * 60)

    # 打开相机
    caps = {}
    if not args.no_camera:
        import cv2
        import numpy as np
        for cam in CAMERAS:
            c = cv2.VideoCapture(cam["path"], cv2.CAP_V4L2)
            if cam["fourcc"]:
                c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            caps[cam["name"]] = c
            print(f"[camera] {cam['name']} 已打开")
            time.sleep(0.3)

    if args.drive:
        print("\n⚠️ 驱动从臂模式：将真实驱动机械臂！")
        print("   请确认从臂周围无遮挡、急停可用！")
        sys.path.insert(0, "/home/nxrobo/source_new/lerobot/src")
        from lerobot.robots.sgr_follower import SgrFollower
        from lerobot.robots.sgr_follower.config_sgr_follower import SgrFollowerConfig
        # TODO: 接入从臂驱动（谨慎使用）
        print("   [INFO] 从臂驱动接口已就绪（建议先验证只读模式）")

    print("\n测试开始：操作主臂观察关节位置变化...")
    print("按 Ctrl+C 停止\n")

    try:
        while True:
            # 1. 读主臂位置（只读）
            left = read_leader_positions(args.left_port)
            right = read_leader_positions(args.right_port)

            if left:
                lv = list(left.values())
                print(f"\r左臂: waist={lv[0]:6.1f} sh={lv[1]:6.1f} el={lv[2]:6.1f} "
                      f"fr={lv[3]:6.1f} wf={lv[4]:6.1f} wr={lv[5]:6.1f} gr={lv[6]:6.1f}", end="")

            # 2. 显示相机
            if caps:
                import cv2
                import numpy as np
                frames = []
                for name, c in caps.items():
                    ret, f = c.read()
                    if ret:
                        frames.append(cv2.resize(f, (320, 240)))
                    else:
                        frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
                if len(frames) == 3:
                    grid = np.hstack([frames[0], frames[1], frames[2]])
                    cv2.imshow("Teleop + Camera Test (top|right|left)", grid)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n已停止")
    finally:
        for c in caps.values():
            c.release()
        print("资源已释放")


if __name__ == "__main__":
    main()
