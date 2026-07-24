#!/usr/bin/env python3
"""
ACT 策略验证 — PushT 环境
绕过 LeRobotDataset（需 HF auth），直接加载本地模型，
在 gym_pusht 环境中验证推理输出。
"""
import sys, json, warnings
import numpy as np
import torch
from pathlib import Path
warnings.filterwarnings("ignore")

MODEL_PATH = Path("/workspace/models/act_pusht")

# ── 1. 加载 ACT 模型 ───────────────────────────────────
def load_act_model(model_dir, device="cuda"):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    cfg = ACTConfig.from_pretrained(str(model_dir))
    policy = ACTPolicy(cfg)
    sd = torch.load(model_dir / "model.safetensors", map_location=device)
    policy.load_state_dict(sd, strict=False)
    policy.to(device).eval()
    n = sum(p.numel() for p in policy.parameters())
    print(f"✅ ACT 加载成功: {n:,} params, device={device}")
    return policy, cfg

# ── 2. PushT 环境 ──────────────────────────────────────
def make_env():
    import gymnasium as gym
    import gym_pusht
    return gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")

# ── 3. 推理 ────────────────────────────────────────────
def infer(policy, obs_img, obs_state, device):
    """obs_img: (96,96,3) uint8, obs_state: (2,) agent_pos"""
    img = obs_img.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    st_t = torch.from_numpy(obs_state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = policy({"observation.state": st_t, "observation.images.top": img_t})
    return out[0, 0, :].cpu().numpy()  # first step of chunk

# ── 4. 主程序 ──────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  设备: {device} | GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")

    policy, cfg = load_act_model(MODEL_PATH, device)
    env = make_env()

    # 4a. 多轮推理验证
    N_TESTS = 10
    print(f"\n{'='*60}")
    print(f"📊 推理验证: {N_TESTS} 次不同初始状态")
    print(f"{'='*60}")

    results = []
    for i in range(N_TESTS):
        obs, _ = env.reset(seed=1000 + i)
        state = obs["agent_pos"] if isinstance(obs, dict) else np.zeros(2)  # fallback
        img = env.render()
        # 需要 resize 到 96x96
        from PIL import Image
        img_pil = Image.fromarray(img).resize((96, 96))
        img_arr = np.array(img_pil)

        action = infer(policy, img_arr, state, device)
        norm = float(np.linalg.norm(action))
        has_nan = bool(np.any(np.isnan(action)))
        has_inf = bool(np.any(np.isinf(action)))
        ok = norm > 0.001 and not has_nan and not has_inf

        results.append({"idx": i, "norm": norm, "ok": ok,
                        "action": action.tolist()})
        tag = "✅ OK" if ok else "❌ FAIL"
        print(f"  [{i:2d}] ||a||={norm:.4f}  action=[{action[0]:+.4f}, {action[1]:+.4f}]  {tag}")

    n_ok = sum(r["ok"] for r in results)
    norms = [r["norm"] for r in results]
    print(f"\n  均值 ||a||={np.mean(norms):.4f} ± {np.std(norms):.4f}")
    print(f"  NaN: {sum(1 for r in results if not r['ok'])}  |  通过: {n_ok}/{N_TESTS}")

    # 4b. 五分钟闭环测试
    print(f"\n{'='*60}")
    print(f"🔄 闭环测试: 5 episodes × 100 steps")
    print(f"{'='*60}")

    for ep in range(5):
        obs, _ = env.reset(seed=2000 + ep)
        total_dist = 0.0
        block_moved = 0.0
        prev_block = None

        for step in range(100):
            state = obs["agent_pos"] if isinstance(obs, dict) else np.zeros(2)
            img = env.render()
            img_pil = Image.fromarray(img).resize((96, 96))
            img_arr = np.array(img_pil)

            action = infer(policy, img_arr, state, device)
            obs, reward, terminated, truncated, info = env.step(action)

            # 追踪 T 块位移
            if "block_pose" in info:
                block_pos = np.array(info["block_pose"][:2])
                if prev_block is not None:
                    block_moved += np.linalg.norm(block_pos - prev_block)
                prev_block = block_pos

            total_dist += float(np.linalg.norm(action))
            if terminated or truncated:
                break

        coverage = float(np.clip(np.sum(obs[-2:]), 0, 1)) if len(obs) > 2 else 0.0
        print(f"  Ep {ep}: steps={step+1:3d}  "
              f"Σ||a||={total_dist:.1f}  block_moved={block_moved:.0f}px  "
              f"coverage={coverage:.2%}")

    env.close()

    # 4c. 保存结果
    output = {
        "model": str(MODEL_PATH),
        "device": device,
        "n_params": sum(p.numel() for p in policy.parameters()),
        "inference_tests": [{"idx": r["idx"], "norm": r["norm"], "ok": r["ok"]} for r in results],
        "inference_pass_rate": f"{n_ok}/{N_TESTS}",
        "mean_action_norm": float(np.mean(norms)),
        "std_action_norm": float(np.std(norms)),
    }
    with open("/workspace/act_verify_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📁 结果已保存: /workspace/act_verify_results.json")

if __name__ == "__main__":
    main()
