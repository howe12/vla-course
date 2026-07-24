#!/usr/bin/env python3
"""物理成功率评估演示 — pick-and-place 章节教学脚本。
对比 legacy_success（纯几何）与 physical_success（四阶段物理约束）。"""
import numpy as np

TARGET = np.array([0.35, 0.10, 0.05])   # 目标位置
TABLE_Z = 0.05                           # 桌面高度
N = 200                                  # 每 episode 帧数

# ── 模拟数据：真阳性 / 假阳性 / 临界 ───────────────────────
def gen_true_positive():
    """Case 1: 真阳性 — 完整抓取→搬运→放置→直立"""
    t = np.linspace(0, 1, N)
    gs, ge, ps = 0.15, 0.35, 0.80
    cup_z = np.piecewise(t,
        [t < gs, (t >= gs) & (t < ge), (t >= ge) & (t < ps), t >= ps],
        [TABLE_Z, lambda x: TABLE_Z + 0.15*(x-gs)/(ge-gs),
         TABLE_Z + 0.15, lambda x: TABLE_Z + 0.15 - 0.07*(x-ps)/(1-ps)])
    cup_xy = np.outer(np.clip((t - 0.2) / 0.6, 0, 1), TARGET[:2])
    cup_pos = np.column_stack([cup_xy, cup_z])
    cup_angle = 3.0 * np.exp(-5 * t)
    ee_pos = cup_pos + np.random.normal(0, 0.008, cup_pos.shape)
    gripper = np.where(t > 0.30, np.random.uniform(0.01, 0.04, N), 0.35)
    return cup_pos, ee_pos, gripper, cup_angle

def gen_false_positive():
    """Case 2: 假阳性 — 夹爪未闭合，杯子根本没动"""
    t = np.linspace(0, 1, N)
    cup_pos = np.column_stack([np.zeros((N, 2)), np.full(N, TABLE_Z)])
    cup_angle = np.zeros(N)
    ee_pos = np.outer(t, TARGET) + np.random.normal(0, 0.005, (N, 3))
    gripper = np.full(N, 0.35)            # 始终张开
    return cup_pos, ee_pos, gripper, cup_angle

def gen_borderline():
    """Case 3: 临界 — 前三阶段通过，直立倾斜 16.5° 超标"""
    t = np.linspace(0, 1, N)
    gs, ps = 0.18, 0.78
    cup_z = np.piecewise(t, [t < gs, (t >= gs) & (t < ps), t >= ps],
        [TABLE_Z, TABLE_Z + 0.12, lambda x: TABLE_Z + 0.12 - 0.04*(x-ps)/(1-ps)])
    cup_xy = np.outer(np.clip((t - 0.2) / 0.6, 0, 1), TARGET[:2])
    cup_pos = np.column_stack([cup_xy, cup_z])
    cup_angle = np.full(N, 5.0)
    cup_angle[-30:] = np.linspace(5, 16.5, 30)  # 末端超标
    ee_pos = cup_pos + np.random.normal(0, 0.008, cup_pos.shape)
    gripper = np.where(t > 0.30, np.random.uniform(0.02, 0.05, N), 0.30)
    return cup_pos, ee_pos, gripper, cup_angle

# ── 四阶段检查 ───────────────────────────────────────────
def check_grasp(cup_z, gripper):
    """抓取：Z > 0.08 且夹爪闭合 < 0.05"""
    lifted, closed = bool(np.any(cup_z > 0.08)), bool(np.any(gripper < 0.05))
    ok = lifted and closed
    return ok, {"lifted": lifted, "closed": closed, "ok": ok}

def check_transport(cup_z, grasp_idx):
    """搬运：抓取后 Z 始终 > 0.07"""
    if grasp_idx is None: return False, {"min_z": 0.0, "ok": False}
    min_z = float(np.min(cup_z[grasp_idx:]))
    ok = min_z > 0.07
    return ok, {"min_z": min_z, "ok": ok}

def check_place(cup_xy):
    """放置：终点 XY 距目标 < 0.05 m"""
    dist = float(np.linalg.norm(cup_xy[-1] - TARGET[:2]))
    ok = dist < 0.05
    return ok, {"dist": dist, "ok": ok}

def check_upright(angle):
    """直立：最终倾角 < 15°"""
    fa = float(np.abs(angle[-1]))
    ok = fa < 15.0
    return ok, {"angle": fa, "ok": ok}

def physical_success(cup_pos, gripper, cup_angle):
    """四阶段物理成功：抓取→搬运→放置→直立"""
    cz, cxy = cup_pos[:, 2], cup_pos[:, :2]
    g_ok, gi = check_grasp(cz, gripper)
    mask = (cz > 0.08) & (gripper < 0.05)
    idx = int(np.argmax(mask)) if np.any(mask) else None
    t_ok, ti = check_transport(cz, idx)
    p_ok, pi = check_place(cxy)
    u_ok, ui = check_upright(cup_angle)
    return all([g_ok, t_ok, p_ok, u_ok]), {
        "grasp": gi, "transport": ti, "place": pi, "upright": ui}

def legacy_success(ee_pos):
    """传统评估：EE 终点距目标 < 0.1 m"""
    return float(np.linalg.norm(ee_pos[-1] - TARGET)) < 0.10

# ── 主程序 ───────────────────────────────────────────────
def main():
    cases = [
        ("Case 1 — 真阳性 (True Positive)",     gen_true_positive),
        ("Case 2 — 假阳性 (False Positive)",     gen_false_positive),
        ("Case 3 — 临界情况 (Borderline)",        gen_borderline),
    ]
    for label, gen in cases:
        cup_pos, ee_pos, gripper, cup_angle = gen()
        leg = legacy_success(ee_pos)
        phy, det = physical_success(cup_pos, gripper, cup_angle)
        d_ee = float(np.linalg.norm(ee_pos[-1] - TARGET))
        print(f"\n{'='*50}\n📋 {label}\n{'='*50}")
        print(f"📍 传统几何 (legacy):   {'✅ 通过' if leg else '❌ 失败'}  "
              f"(EE→target={d_ee:.4f} m)")
        print(f"🔬 物理成功 (physical): {'✅ 通过' if phy else '❌ 失败'}")
        stages = [
            ("🤖 抓取 Grasp",     det["grasp"],     ("杯子离桌","lifted"),("夹爪闭合","closed")),
            ("🚚 搬运 Transport", det["transport"],  ("最低高度","min_z")),
            ("📍 放置 Place",     det["place"],      ("终点距离","dist")),
            ("🧍 直立 Upright",   det["upright"],    ("倾斜角度","angle")),
        ]
        for name, info, *pairs in stages:
            vals = ", ".join(
                f"{cn}={info.get(k, 'N/A'):.4f}" if isinstance(info.get(k), float)
                else f"{cn}={info.get(k, 'N/A')}" for cn, k in pairs)
            print(f"    {name}: {'✅ OK' if info.get('ok') else '⚠️  FAIL'}  [{vals}]")
        print()

if __name__ == "__main__":
    main()
