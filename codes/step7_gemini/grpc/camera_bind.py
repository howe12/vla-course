#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_bind.py — LEO-Gemini 四相机绑定向导 (Orin NX / JetPack 6.2)

目的: 确定每个采集节点 (/dev/videoN) 对应的物理相机槽位 (top/front/left/right),
      生成 cameras.json 配置, 并可选生成/安装 udev 稳定软链接 (/dev/camera_<slot>)。

用法 (在 Orin 上, 已激活 venv):
  python camera_bind.py --list                              # 只打印相机清单 (安全, 只读)
  python camera_bind.py                                     # 交互式绑定向导 (推荐, 需人在机器旁挥手)
  python camera_bind.py --slots top,front,left,right        # 自定义槽位顺序 (默认即此)
  python camera_bind.py --apply-udev                        # 生成并用 sudo 安装 udev 规则
  python camera_bind.py --assign top=0,front=6,left=2,right=4   # 脚本化指派 (供自动化/测试)

交互流程 (每槽位):
  1) [m] 挥手检测 — 走到对应物理相机前挥手 ~3 秒, 脚本从多路画面中自动识别是哪一路 (推荐)
  2) [i] 手动输入清单中的编号
  3) [s] 跳过 (保持未分配)

⚠️ 已知硬件限制 (USB 带宽, 已实测):
  - USB2 总线上的 3 台 Microdia 相机最多同时采集 2 路 (第 3 路会被饿死, 0 帧)
  - USB3 相机 (top 位) 不受影响, 可与任意 2 路 USB2 相机同时采集
  - 因此挥手识别采用"分组扫描": 候选 >2 台 USB2 相机会自动分两轮,
    最多挥两次手即可定位一台相机; 识别出的相机越多, 后续轮次越少

⚡ 安全: 本脚本只读相机, 不发任何电机指令。
⚠️ 同时只运行一个采集程序。
"""

import os
import sys
import glob
import json
import time
import re
import argparse
import subprocess
import datetime
import getpass

# OpenCV / GStreamer 静音, 减少警告噪音
os.environ.setdefault("GST_DEBUG", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_OUT = os.path.join(WORKSPACE_DIR, "cameras.json")
DEFAULT_RULES = os.path.join(WORKSPACE_DIR, "udev", "99-camera-bind.rules")
DEFAULT_SLOTS = ["top", "front", "left", "right"]

# 槽位 → 物理位置提示
SLOT_HINT = {
    "top":    "左上·全局俯视",
    "front":  "右上·平视视角",
    "left":   "左下·左臂",
    "right":  "右下·右臂",
}


def log(msg=""):
    print(msg, flush=True)


def err(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 设备发现

def discover_devices():
    """枚举 /dev/video* 并关联 by-path / udev 属性。"""
    video_glob = sorted(
        glob.glob("/dev/video[0-9]*"),
        key=lambda p: int(re.search(r"(\d+)$", p).group(1)),
    )
    bypath_map = {}
    for link in glob.glob("/dev/v4l/by-path/*"):
        bypath_map[os.path.realpath(link)] = os.path.basename(link)

    devs = []
    for path in video_glob:
        props = {}
        try:
            out = subprocess.run(
                ["udevadm", "info", "-q", "property", "-n", path],
                capture_output=True, text=True, timeout=5,
            ).stdout
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    props[k] = v
        except Exception:
            pass
        caps = props.get("ID_V4L_CAPABILITIES", "")
        devs.append({
            "video": path,
            "index": int(re.search(r"(\d+)$", path).group(1)),
            "by_path": bypath_map.get(os.path.realpath(path)),
            "id_path": props.get("ID_PATH", ""),
            "bus_path": props.get("DEVPATH", ""),
            "vendor_id": props.get("ID_VENDOR_ID", ""),
            "product_id": props.get("ID_MODEL_ID", ""),
            "model": props.get("ID_MODEL", "") or props.get("ID_V4L_PRODUCT", ""),
            "serial": props.get("ID_SERIAL_SHORT", ""),
            "capture": "capture" in caps,
            "open": False,
            "probe": None,
            # 是否挂在 USB2 总线上 (kernel DEVPATH 含 /usb1/; USB3 为 /usb2/)
            "usb2_bus": "/usb1/" in props.get("DEVPATH", ""),
        })
    return devs


def probe_devices(devs):
    """对采集节点做只读打开 + 640x480 单帧探测。"""
    for d in devs:
        if not d["capture"]:
            continue
        cap = cv2.VideoCapture(d["video"])
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(5):            # 刷掉缓冲帧
            cap.read()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        if ret and frame is not None:
            d["open"] = True
            d["probe"] = f"{frame.shape[1]}x{frame.shape[0]}"
        cap.release()


def print_inventory(devs, numbered=False):
    rows = [d for d in devs if d["capture"]]
    log("  # | 设备        | 状态        | by-path (USB 端口)                  | 厂商:型号")
    log("----+-------------+-------------+-------------------------------------+----------------------")
    for i, d in enumerate(rows):
        num = f"{i:2d}" if numbered else "--"
        ok = f"OK {d['probe']}" if d["open"] else "CLOSED"
        ident = f"{d['vendor_id']}:{d['product_id']} {d['model']}".strip()
        log(f" {num:>2} | {d['video']:<11} | {ok:<11} | {str(d['by_path'] or d['bus_path'])[:37]:<37} | {ident}")
    if not rows:
        err("!! 未发现任何可用的采集设备 (/dev/videoN)。请检查 USB 是否插好: lsusb")
    return rows


# ---------------------------------------------------------------- 动作识别

def _open_cap(path):
    """打开相机并尽量用 MJPG + 低分辨率 (降低 USB2 带宽压力)。"""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    return cap


def motion_detect(devs, duration=3.0, live=True):
    """同时打开多路采集 (调用方需保证 ≤2 台 USB2 相机), 找出运动最大的那一路。

    返回 (winning_device | None, {video: avg_diff})
    """
    caps, prev, sums, counts = {}, {}, {}, {}
    for d in devs:
        cap = _open_cap(d["video"])
        if not cap.isOpened():
            err(f"   !! 无法打开 {d['video']}, 已跳过")
            cap.release()
            continue
        caps[d["video"]] = cap
        sums[d["video"]] = 0.0
        counts[d["video"]] = 0

    if not caps:
        return None, {}

    for _ in range(5):                 # 预热, 排掉旧帧
        for cap in caps.values():
            cap.read()

    t0 = time.time()
    last_print = 0.0
    try:
        while time.time() - t0 < duration:
            for v, cap in list(caps.items()):
                ret, frame = cap.read()
                if not ret or frame is None:
                    err(f"   !! {v} 读取失败 (带宽不足?), 已剔除")
                    cap.release()
                    del caps[v]
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if v in prev:
                    sums[v] += float(cv2.absdiff(gray, prev[v]).mean())
                    counts[v] += 1
                prev[v] = gray
            if live and time.time() - last_print >= 0.5:
                scores = {
                    v: (sums[v] / counts[v] if counts[v] else 0.0)
                    for v in caps
                }
                line = "  " + "  ".join(
                    f"{os.path.basename(v)}:{s:5.1f}" for v, s in sorted(scores.items())
                )
                log("\r" + line + " " * 12)
                last_print = time.time()
    except KeyboardInterrupt:
        pass
    for cap in caps.values():
        cap.release()
    log("")

    avg = {v: sums[v] / counts[v] for v in sums if counts.get(v)}
    if not avg:
        return None, {}
    winner_v = max(avg, key=avg.get)
    if avg[winner_v] < 5.0:            # 低于阈值视为无运动 (环境底噪 ~2.7, 挥手通常 >10)
        return None, avg
    winner = next((d for d in devs if d["video"] == winner_v), None)
    return winner, avg


def identify_slot(devs, duration=3.0, live=True):
    """对一组未分配相机识别一个槽位 (自动规避 USB2 三路并发饿死问题)。

    策略:
      - USB2 候选 ≤2 台 (或不区分总线): 单轮扫描全部
      - USB2 候选 ≥3 台: 分两轮 (每轮 ≤2 台 USB2 + 可选的 USB3 台),
        第一轮没命中自动补看遗漏的, 最多两轮必能定位
    返回 (winner_device | None, avg, rounds_used)
    """
    cands = list(devs)
    usb2 = [d for d in cands if d.get("usb2_bus")]
    usb3 = [d for d in cands if not d.get("usb2_bus")]
    rounds = []
    if len(usb2) >= 3:
        rounds.append(usb3 + usb2[:2])
        rounds.append(usb3 + usb2[2:])
    else:
        rounds.append(cands)

    for gi, group in enumerate(rounds, start=1):
        shown = " ".join(d["video"] for d in group)
        if gi == 1:
            log(f"  → 本轮同时看: {shown}")
        else:
            log(f"  → 第一轮没命中, 补看遗漏的: {shown}, 请再挥手一次")
        winner, avg = motion_detect(group, duration=duration, live=live)
        if winner is not None:
            return winner, avg, gi
    return None, {}, len(rounds)


def dev_info(d):
    """提取用于配置/规则的设备叶子字段 (无 Host 引用, 纯数据)。"""
    return {
        "video": d["video"],
        "by_path": d["by_path"],
        "id_path": d["id_path"],
        "bus_path": d["bus_path"],
        "vendor_id": d["vendor_id"],
        "product_id": d["product_id"],
        "model": d["model"],
        "serial": d["serial"],
    }


# ---------------------------------------------------------------- 交互向导

def interactive_bind(devs, slots, detect_duration):
    captures = [d for d in devs if d["capture"]]
    assigned, used = {}, {}

    def menu(slot):
        hint = SLOT_HINT.get(slot, "")
        log("")
        log(f"========== 识别槽位 [{slot}] ({hint}) ==========")
        log("  1) [m] 挥手检测 —— 到该物理相机镜头前挥手几秒, 自动识别")
        log("  2) [i] 手动输入编号")
        log("  3) [s] 跳过")
        return input(f"  槽位 '{slot}' 选择 [m/i/s]: ").strip().lower()

    for slot in slots:
        if slot in assigned:
            continue
        while True:
            choice = menu(slot)
            if choice.startswith("s"):
                log(f"  - 跳过 '{slot}'")
                break
            if choice.startswith("m"):
                remain = [d for d in captures if d["video"] not in used]
                if not remain:
                    log("  !! 没有剩余相机可供识别")
                    break
                log(f"  → 剩余 {len(remain)} 台未绑定: "
                    + ", ".join(d["video"] for d in remain))
                log(f"  → 请到 '{slot}' 相机前挥手 ~{int(detect_duration)} 秒 ...")
                winner, avg, rounds = identify_slot(remain, duration=detect_duration)
                if winner is None:
                    err("  !! 未检测到明显动作 (阈值 3.0), 可重试或手动指定")
                    continue
                log(f"  → 检测到最大运动: {winner['video']} "
                    f"(by-path {winner['by_path']}, 平均帧差 {avg.get(winner['video'], 0):.1f}, 用了 {rounds} 轮)")
                confirm = input(f"  确认槽位 '{slot}' = {winner['video']} ? [y/N]: ").strip().lower()
                if confirm != "y":
                    log("  - 重新识别")
                    continue
                assigned[slot] = winner
                used[winner["video"]] = slot
            elif choice.startswith("i"):
                rows = print_inventory(devs, numbered=True)
                try:
                    pick = int(input("  输入清单中的编号: ").strip())
                    d = rows[pick]
                except (ValueError, IndexError):
                    err("  !! 编号无效")
                    continue
                if d["video"] in used:
                    err(f"  !! {d['video']} 已分配给 '{used[d['video']]}' 槽位")
                    continue
                assigned[slot] = d
                used[d["video"]] = slot
            else:
                err("  !! 请输入 m / i / s")
                continue
            break
        log(f"  ✔ 已绑定: {slot} -> {assigned[slot]['video']} "
            f"({assigned[slot]['by_path']})")

    unassigned = [d for d in captures if d["video"] not in used]
    return assigned, unassigned


# ---------------------------------------------------------------- 输出

def udev_rule_text(assigned):
    lines = [
        "# generated by camera_bind.py on " + datetime.date.today().isoformat(),
        "# 按 USB 物理端口绑定采集节点 (capture, index0), 元数据节点不受影响",
        "# 生效后: /dev/camera_<slot> -> capture 节点",
        "",
    ]
    for slot in DEFAULT_SLOTS:
        d = assigned.get(slot)
        if not d or not d["id_path"]:
            lines.append(f"# (slot '{slot}' 未分配或缺少 ID_PATH, 规则未生成)")
            continue
        lines.append(
            f'SUBSYSTEM=="video4linux", KERNEL=="video[0-9]*", '
            f'ENV{{ID_V4L_CAPABILITIES}}==":capture:", '
            f'ENV{{ID_PATH}}=="{d["id_path"]}", '
            f'SYMLINK+="camera_{slot}"'
        )
    return "\n".join(lines) + "\n"


def write_outputs(assigned, unassigned, devs, out_json, rules_path):
    payload = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "device_count": len([d for d in devs if d["capture"]]),
        "slots": {s: dev_info(d) for s, d in assigned.items()},
        "unassigned_capture_devices": [dev_info(d) for d in unassigned],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"✔ 配置已写入: {out_json}")

    if rules_path:
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)
        with open(rules_path, "w", encoding="utf-8") as f:
            f.write(udev_rule_text(assigned))
        log(f"✔ udev 规则已生成: {rules_path}  (--apply-udev 安装到系统)")

    log("")
    log("=== 当前绑定汇总 (供 camera_live_4x.py 对照) ===")
    for slot in DEFAULT_SLOTS:
        d = assigned.get(slot)
        if d:
            log(f"  {slot:<6} -> {d['video']:<10} {d['by_path'] or ''}")
    for d in unassigned:
        log(f"  (未分配)    -> {d['video']:<10} {d['by_path'] or ''}")
    log("")


def apply_udev(rules_path):
    if not os.path.exists(rules_path):
        err(f"!! 规则文件不存在: {rules_path}")
        return False
    log("需要 sudo 密码以安装 udev 规则 ...")
    pw = getpass.getpass("sudo 密码: ")

    def sudorun(args):
        r = subprocess.run(
            ["sudo", "-S", "--"] + args,
            input=pw + "\n", text=True, capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            err("sudo 失败: " + r.stderr.strip())
            raise RuntimeError("sudo failed")
        return r

    try:
        sudorun(["-v"])                              # 验证并缓存凭据
        sudorun(["install", "-m", "0644", rules_path, "/etc/udev/rules.d/"])
        sudorun(["udevadm", "control", "--reload-rules"])
        sudorun(["udevadm", "trigger", "--subsystem-match=video4linux"])
    except RuntimeError:
        err("!! 规则未安装。请检查密码后重试。")
        return False
    log("✔ udev 规则已安装并 reload/trigger 完成。")
    log("  检查: ls -l /dev/camera_*   (若为空, 拔插一次 USB 相机)")
    return True


# ---------------------------------------------------------------- 入口

def parse_assign(spec, captures):
    """'top=0,left=2,...' → {slot: dev}"""
    by_index = {d["index"]: d for d in captures}
    mapping = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            err(f"!! 无法解析 '{item}' (应为 slot=编号)")
            return None
        slot, idx = item.split("=", 1)
        try:
            idx = int(idx)
        except ValueError:
            err(f"!! 编号无效: {idx}")
            return None
        if idx not in by_index:
            err(f"!! 编号 {idx} 没有对应的采集设备 (可用: "
                + ", ".join(str(d["index"]) for d in captures) + ")")
            return None
        if slot in mapping:
            err(f"!! 槽位 '{slot}' 重复")
            return None
        mapping[slot] = by_index[idx]
    return mapping


def main():
    ap = argparse.ArgumentParser(
        description="LEO-Gemini 四相机绑定向导",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="交互流程: 每槽位可选 [m]挥手检测 / [i]手动编号 / [s]跳过",
    )
    ap.add_argument("--list", action="store_true", help="只打印相机清单")
    ap.add_argument("--slots", default=",".join(DEFAULT_SLOTS),
                    help="槽位顺序, 默认 top,front,left,right")
    ap.add_argument("--assign", metavar="SLOT=N,...",
                    help="非交互指派, 如 top=0,front=6,left=2,right=4")
    ap.add_argument("--out", default=DEFAULT_OUT, help="JSON 输出路径")
    ap.add_argument("--udev", default=DEFAULT_RULES,
                    help="udev 规则输出路径 (空=不生成)")
    ap.add_argument("--apply-udev", action="store_true",
                    help="生成规则后用 sudo 安装到系统")
    ap.add_argument("--detect-duration", type=float, default=3.0,
                    help="挥手检测窗口秒数, 默认 3")
    args = ap.parse_args()

    if args.slots:
        slots = [s.strip().lower() for s in args.slots.split(",") if s.strip()]
    else:
        slots = list(DEFAULT_SLOTS)

    devs = discover_devices()
    probe_devices(devs)
    captures = [d for d in devs if d["capture"]]

    log("==== 相机枚举 (只读探测, 安全) ====")
    rows = print_inventory(devs, numbered=False)
    if len(captures) != len(DEFAULT_SLOTS):
        log(f"⚠ 发现 {len(captures)} 路采集设备, 期望 {len(DEFAULT_SLOTS)} 路。")

    if args.list:
        return 0

    if args.assign:
        mapping = parse_assign(args.assign, captures)
        if mapping is None:
            return 2
        assigned = {s: d for s, d in mapping.items() if s in slots}
        extra = {s: d for s, d in mapping.items() if s not in slots}
        if extra:
            log("以下槽位不在 --slots 中, 仍写入 JSON: " + ", ".join(extra))
            assigned.update(extra)
        used = {d["video"] for d in assigned.values()}
        unassigned = [d for d in captures if d["video"] not in used]
        write_outputs(assigned, unassigned, devs, args.out,
                      args.udev if args.udev else None)
    else:
        assigned, unassigned = interactive_bind(devs, slots, args.detect_duration)
        write_outputs(assigned, unassigned, devs, args.out,
                      args.udev if args.udev else None)

    if args.apply_udev and args.udev:
        apply_udev(args.udev)

    log("完成。下一步建议: python camera_live_4x.py --mode 3a 验证 4 路实时预览")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        err("\n已中断。未写入任何配置。")
        sys.exit(130)
    except EOFError:
        err("\n!! 输入流已结束 (非交互环境请用 --assign 或 --list)")
        sys.exit(2)