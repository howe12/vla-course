#!/usr/bin/env python3
"""Sagittarius SGR532 键盘遥操作 (teleop) + 夹爪控制 + 轨迹记录 (02-4 / 数据采集)

交互模式(有显示, 本机)   : python sgr532_teleop.py
无头自检(记录链路)        : python sgr532_teleop.py --demo

键位:
  q/a   J1 ±      w/s  J2 ±      e/d  J3 ±
  r/f   J4 ±      t/g  J5 ±      y/h  J6 ±
  z     夹爪张开    c   夹爪闭合
  空格  开始/停止 录制
  k     保存当前轨迹到 CSV
  Esc   退出
"""
import os, sys, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import mujoco

m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "sgr532_motion.mjcf"))
d = mujoco.MjData(m)
assert m.nu == 8

J = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"J{i}") for i in range(1, 7)]
GL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "GL")
GR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "GR")
qadr = m.jnt_qposadr[[*J, GL, GR]]
JNAME = ["J1", "J2", "J3", "J4", "J5", "J6", "GL", "GR"]

CTRL = np.zeros(8)          # 目标关节角 (rad)
REC = {"on": False, "rows": [], "t0": 0.0}   # 录制状态

STEP = 0.06                 # 每次按键的关节增量 (rad)
GRIP_OPEN, GRIP_CLOSE = 0.008, -0.008

def clamp():
    pass

def save_traj():
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(HERE, f"sgr532_traj_{ts}.csv")
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + JNAME)
        for t, q in REC["rows"]:
            w.writerow([f"{t:.6f}"] + [f"{v:.6f}" for v in q])
    print(f"[存储] {len(REC['rows'])} 步 -> {fn}")
    return fn

def key_callback(keycode):
    ch = chr(keycode) if 0 <= keycode < 128 else ""
    if ch in "qwasde":  # J1 J2 J3
        jmap = {"q": 0, "w": 1, "e": 2, "a": 0, "s": 1, "d": 2}
        CTRL[jmap[ch.lower()] if ch.lower() in jmap else 0] += STEP if ch in "qwe" else -STEP
    elif ch in "rftygh":
        jmap = {"r": 3, "t": 4, "y": 5, "f": 3, "g": 4, "h": 5}
        CTRL[jmap[ch.lower()]] += STEP if ch in "rty" else -STEP
    elif ch == "z":
        CTRL[6] = GRIP_OPEN; CTRL[7] = GRIP_OPEN
        print("[夹爪] 张开")
    elif ch == "c":
        CTRL[6] = GRIP_CLOSE; CTRL[7] = GRIP_CLOSE
        print("[夹爪] 闭合")
    elif ch == " ":
        REC["on"] = not REC["on"]
        if REC["on"]:
            REC["rows"] = []; REC["t0"] = d.time
            print("[录制] 开始")
        else:
            print(f"[录制] 停止, 共 {len(REC['rows'])} 步 (按 k 保存)")
    elif ch == "k":
        save_traj()
    elif ch == "?":
        help_txt()

def help_txt():
    print("键位 | QJ1+ AJ1- WJ2+ SJ2- EJ3+ DJ3- | RJ4+ FJ4- TJ5+ GJ5- YJ6+ HJ6-")
    print("     | Z 夹爪张开   C 夹爪闭合   空格 录制开/停   K 保存轨迹   Esc 退出")

def demo():
    """无头自检: 模拟一段遥操作, 验证‘伺服 + 录制 + 存CSV’链路"""
    print("[demo] 无显示自检, 模拟夹爪开合 + 关节角度轨迹录制")
    d.qpos[list(qadr)] = np.zeros(8); mujoco.mj_forward(m, d)
    seq = [  # 模拟一段"下探-闭合-抬-开-回" 的遥操作序列 (逐帧目标)
        np.array([0.0, -0.9, 0.8, 0.4, -0.5, 0.6, 0.008, 0.008]),
        np.array([0.3, 0.5, -0.6, -0.4, 0.5, 0.3, -0.008, -0.008]),
        np.array([0.4, -1.0, 1.2, 0.7, -0.7, 0.9, -0.008, -0.008]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.008, 0.008]),
    ]
    REC["on"] = True; REC["t0"] = d.time
    for target in seq:
        for _ in range(300):
            d.ctrl[:] = target
            mujoco.mj_step(m, d)
            REC["rows"].append((d.time - REC["t0"], d.qpos[list(qadr)].copy()))
    REC["on"] = False
    fn = save_traj()
    print("[demo] OK, 轨迹样本:")
    import itertools
    for t, q in REC["rows"][::60]:
        line = " ".join(f"{v:7.3f}" for v in q)
        print(f"  t={t:6.3f}  qpos={line}")
    print("=== SGR532 teleop 记录链路验证通过 ===")

def run_terminal():
    """无窗口键盘遥操作: 任何终端可跑(不需 mujoco.viewer/显示), 每步伺服+打印关节角+可选录制"""
    import select
    is_tty = sys.stdin.isatty()
    old = None
    if is_tty:
        import tty, termios
        old = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    print("[terminal] 键盘遥操作(无窗口). 键位同交互版, Ctrl-C 退出")
    try:
        cnt = 0
        while True:
            if select.select([sys.stdin], [], [], 0.03)[0]:
                ch = sys.stdin.read(1)
                if not ch or ch in ("\x03", "\x1b"):
                    break
                key_callback(ord(ch))
            d.ctrl[:] = CTRL
            mujoco.mj_step(m, d)
            if REC["on"]:
                REC["rows"].append((d.time - REC["t0"], d.qpos[list(qadr)].copy()))
            cnt += 1
            if cnt % 8 == 0:
                q = " ".join(f"{v:6.3f}" for v in d.qpos[list(qadr)])
                print(f"\r qpos[{q}]", end="", flush=True)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if old is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        print("")
    if REC["rows"]:
        save_traj()

if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--terminal" in sys.argv:
        run_terminal()
    else:
        print("[teleop] 启动 MuJoCo 窗口, 键盘控制 Sagittarius SGR532")
        help_txt()
        with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as view:
            while view.is_running():
                d.ctrl[:] = CTRL
                mujoco.mj_step(m, d)
                if REC["on"]:
                    REC["rows"].append((d.time - REC["t0"], d.qpos[list(qadr)].copy()))
                view.sync()