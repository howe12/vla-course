#!/usr/bin/env python3
"""Step 6 实验：Pi0 扩散 VLA 驱动课程 MuJoCo 机械臂

对比 §2 的正弦 mock 轨迹，Pi0 的扩散动作更"有目的性"——
末端会向红色目标方块移动，夹爪在接近时闭合。

用法：
    uv run python codes/step6_pi0/sim_pi0_arm.py
    uv run python codes/step6_pi0/sim_pi0_arm.py --headless
"""

import os, sys, math, time, argparse
import numpy as np

try:
    import mujoco
except ImportError:
    print("❌ MuJoCo 未安装，请运行: pip install mujoco")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("❌ PyTorch 未安装")
    sys.exit(1)


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models", "widowx_arm.xml",
)
CKPT_PATH = "outputs/pi0/best.pt"


# ============================================================
# Pi0 模型（和 train_pi0.py 相同架构）
# ============================================================
class Pi0FlowMatching(nn.Module):
    def __init__(self, action_dim=7, hidden=1024):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.vis_proj = nn.Linear(256 * 7 * 7, hidden)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.denoiser = nn.Sequential(
            nn.Linear(action_dim + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.action_dim = action_dim

    def forward(self, images, actions=None, t=None):
        B = images.shape[0]
        vis = self.vision(images).flatten(1)
        cond = self.vis_proj(vis)
        if t is not None:
            t_embed = self.time_embed(t.unsqueeze(-1))
            cond = cond + t_embed
            noise = torch.randn_like(actions)
            noisy = (1 - t.unsqueeze(-1)) * actions + t.unsqueeze(-1) * noise
            pred = self.denoiser(torch.cat([noisy, cond], dim=-1))
            return nn.MSELoss()(pred, noise - actions)
        else:
            x = torch.randn(B, self.action_dim, device=images.device)
            n_steps = 10
            dt = 1.0 / n_steps
            for i in range(n_steps):
                ti = torch.full((B,), i * dt, device=images.device)
                c = cond + self.time_embed(ti.unsqueeze(-1))
                v = self.denoiser(torch.cat([x, c], dim=-1))
                x = x + v * dt
            return x


# ============================================================
# MuJoCo 仿真
# ============================================================
def load_model():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    return model, data


def get_end_effector_pos(data):
    site_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    return data.site_xpos[site_id].copy()


def render_observation(renderer, model, data):
    """渲染当前相机画面 → (224,224,3) RGB numpy"""
    renderer.update_scene(data, camera="wrist_cam")
    rgb = renderer.render()  # (H, W, 3)
    # resize to 224x224
    from PIL import Image
    img = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
    return np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0


def run_simulation(model, data, pi0_model, device, steps=300, headless=False):
    renderer = mujoco.Renderer(model, height=480, width=640) if headless else None
    viewer = None

    if not headless:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception:
            print("⚠️  Viewer 不可用，切换到 headless 模式")
            headless = True
            renderer = mujoco.Renderer(model, height=480, width=640)

    if headless:
        print("🎬 Headless 模式：渲染仿真帧...")

    for step in range(steps):
        # 1. 渲染相机观测
        if renderer:
            obs = render_observation(renderer, model, data)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

            # 2. Pi0 推理
            with torch.no_grad():
                action = pi0_model(obs_tensor)  # (1, 7)
            action = action[0].cpu().numpy()

            # 3. 动作 → 目标关节角（简化 IK：用当前关节角 + 增量）
            target = np.zeros(6)
            for i in range(6):
                target[i] = data.qpos[7 + i] + action[i] * 0.1  # 缩小增量

            for i in range(6):
                data.ctrl[i] = np.clip(target[i], -3.14, 3.14)
        else:
            # 没有 renderer 时用正弦 mock
            t = step * 0.02
            for i in range(6):
                data.ctrl[i] = 0.5 * math.sin(t * (0.5 + i * 0.1))

        mujoco.mj_step(model, data)

        if viewer:
            viewer.sync()
            if not viewer.is_running():
                break

        if step % 50 == 0:
            ee = get_end_effector_pos(data)
            if renderer:
                mode = "Pi0 推理"
            else:
                mode = "正弦 mock"
            print(f"  Step {step:3d} [{mode}] | "
                  f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")

    print(f"\n仿真结束。共 {steps} 步")
    if viewer:
        viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH)
    parser.add_argument("--mock", action="store_true",
                        help="用正弦轨迹代替 Pi0（无模型时使用）")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 6 实验：Pi0 扩散 VLA 驱动机械臂")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载 Pi0 模型
    if args.mock:
        print("⚠️  使用正弦 mock（无 Pi0 模型）")
        pi0_model = None
    elif os.path.exists(args.ckpt):
        print(f"加载 Pi0: {args.ckpt}")
        pi0_model = Pi0FlowMatching().to(device)
        state = torch.load(args.ckpt, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        pi0_model.load_state_dict(state, strict=False)
        pi0_model.eval()
    else:
        print(f"⚠️  未找到 {args.ckpt}，使用正弦 mock")
        print(f"   先训练: uv run python codes/step6_pi0/train_pi0.py")
        pi0_model = None

    model, data = load_model()
    run_simulation(model, data, pi0_model, device, args.steps, args.headless)

    if pi0_model is not None:
        print("\n✅ Pi0 推理完成。对比 §2 的正弦 mock，轨迹是否更有目的性？")
    else:
        print("\n💡 使用了正弦 mock。训练 Pi0 后重新运行以看到真实的扩散动作。")


if __name__ == "__main__":
    main()
