#!/usr/bin/env python3
"""LEO-Gemini 电机状态只读监测工具

读取双臂（left_arm + right_arm）共 14 个关节的当前位置。
只解析电机主动上报的 SgrMotorsBus 帧（0x55 0xAA），不发送任何写指令，
电机保持当前姿态，绝对安全。

用法（lerobot_061 环境）:
    python3 read_motor_state.py

输出: 各关节位置（0.1° 原始单位 ÷10 = 角度）
依赖: pyserial (pip install pyserial)
"""
import serial
import time
import struct

JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_flex", "wrist_roll", "gripper"]

def parse_frame(data):
    """解析 0x55 0xAA 帧：7 个关节位置（int16 LE / 10）"""
    if len(data) < 3 or data[0] != 0x55 or data[1] != 0xAA:
        return None
    pkt_len = data[2]
    if len(data) < 3 + pkt_len + 2:
        return None
    payload = data[3:3+pkt_len]
    positions = {}
    pos = 1
    for j in JOINTS:
        if pos + 2 <= len(payload):
            val = struct.unpack("<h", payload[pos:pos+2])[0] / 10.0
            positions[j] = val
            pos += 2
    return positions

for port_name, arm_name in [("/dev/ttyACM_left_follower", "left_arm"),
                            ("/dev/ttyACM_right_follower", "right_arm")]:
    print(f"\n=== {arm_name} ({port_name}) ===")
    try:
        s = serial.Serial(port_name, 115200, timeout=0.5)
        buf = b""
        got = False
        t0 = time.time()
        while time.time() - t0 < 3.0 and not got:
            buf += s.read(256)
            idx = buf.find(b"\x55\xaa")
            if idx >= 0 and idx + 3 <= len(buf):
                pkt_len = buf[idx+2]
                if idx + 3 + pkt_len + 2 <= len(buf):
                    frame = buf[idx:idx+3+pkt_len+2]
                    pos = parse_frame(frame)
                    if pos:
                        for j, v in pos.items():
                            print(f"  {j}: {v:8.2f}")
                        got = True
            if idx > 0:
                buf = buf[idx:]
        if not got:
            print("  (3 秒内未解析到完整帧)")
        s.close()
    except Exception as e:
        print(f"  ERROR: {e}")
print("\nDone: read-only, no commands sent.")
