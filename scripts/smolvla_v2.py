#!/usr/bin/env python3
"""SmolVLA Inference — fixed config loading."""
import os, sys, json, argparse, cv2, numpy as np, torch
sys.path.insert(0, "/workspace/vla-course")
from mujoco_env.y_env2 import SimpleEnv2

CKPT_DIR = "/workspace/vla-course/weights/smolvla/weights"

def load_smolvla(checkpoint_dir, device="cuda"):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    from safetensors.torch import load_file

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        cfg = json.load(f)
    for k in ["input_features", "output_features"]:
        if k in cfg and cfg[k]:
            cfg[k] = {kk: PolicyFeature(type=FeatureType[vv["type"]], shape=tuple(vv["shape"])) for kk, vv in cfg[k].items()}
    cfg.pop("type", None); cfg["device"] = device
    valid = {f.name for f in SmolVLAConfig.__dataclass_fields__.values()}
    config = SmolVLAConfig(**{k: v for k, v in cfg.items() if k in valid})
    print("[SmolVLA] Creating policy...")
    policy = SmolVLAPolicy(config)
    print("[SmolVLA] Loading weights...")
    policy.load_state_dict(load_file(os.path.join(checkpoint_dir, "model.safetensors")), strict=False)
    policy.to(device); policy.eval(); policy.reset()
    print(f"[SmolVLA] Loaded on {device}")
    return policy

def load_tokenizer():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True, trust_remote_code=True)

def tokenize(proc, text, max_len=48):
    t = proc.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    return t["input_ids"], t["attention_mask"].bool()

def build_batch(rgb_a, rgb_e, state, lang_t, lang_m, dev="cuda"):
    return {
        "observation.image": torch.from_numpy(np.ascontiguousarray(rgb_a)).float().div(255.0).permute(2,0,1).unsqueeze(0).to(dev),
        "observation.wrist_image": torch.from_numpy(np.ascontiguousarray(rgb_e)).float().div(255.0).permute(2,0,1).unsqueeze(0).to(dev),
        "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(dev),
        "observation.language.tokens": lang_t.to(dev),
        "observation.language.attention_mask": lang_m.to(dev),
    }

def save_video(path, frames, fps=20):
    if not frames: return
    h, w = frames[0].shape[:2]
    wri = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames: wri.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wri.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", default="Place the blue mug on the plate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=CKPT_DIR)
    parser.add_argument("--video", default="/workspace/vla-course/outputs/smolvla_inference/smolvla_inference.mp4")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--xml", default="/workspace/vla-course/models/omy/example_scene_y2.xml")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    policy = load_smolvla(args.checkpoint, args.device)
    processor = load_tokenizer()
    lang_t, lang_m = tokenize(processor, args.instruction)

    env = SimpleEnv2(xml_path=args.xml, action_type='eef_pose', state_type='joint_angle', seed=args.seed)
    env.instruction = args.instruction
    policy.reset()

    # Reset to initial pose
    q_init = np.deg2rad([0, 0, 0, 0, 0, 0])
    env.env.forward(q=q_init, joint_names=env.joint_names, increase_tick=False)
    for _ in range(50): env.step_env()

    frames = []
    success = False
    print(f"\n{'='*60}")
    print(f"[Inference] {args.instruction}  |  seed={args.seed}  |  max_steps={args.max_steps}")
    print(f"{'='*60}")

    for step in range(args.max_steps):
        env.grab_image()
        state = env.get_joint_state()[:6]
        batch = build_batch(env.rgb_agent, env.rgb_ego, state, lang_t, lang_m, args.device)
        action = policy.select_action(batch).squeeze(0).cpu().numpy()
        env.step(action)
        env.step_env()

        # Check success
        obj = "body_obj_mug_5" if "red" in args.instruction.lower() else "body_obj_mug_6"
        try:
            op, _ = env.env.get_pR_body(obj)
            pp, _ = env.env.get_pR_body("body_obj_plate_11")
            if op[2] - pp[2] > 0.03:
                success = True
                print(f"[Step {step+1:3d}] 🎉 SUCCESS!")
                break
        except: pass

        # Render
        agent = cv2.resize(env.rgb_agent, (320, 240))
        ego = cv2.resize(env.rgb_ego, (320, 240))
        comp = np.hstack([agent, ego])
        cv2.putText(comp, args.instruction, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        frames.append(comp)

        if step % 50 == 0:
            print(f"[Step {step+1:3d}] action={action.round(3)}, eef_z={env.p0[2]:.4f}")

    if not args.no_video:
        os.makedirs(os.path.dirname(args.video), exist_ok=True)
        save_video(args.video, frames)
        print(f"\n[Video] {len(frames)} frames → {args.video}")

    print(f"[Result] steps={step+1}/{args.max_steps}  success={'✅' if success else '❌'}")

if __name__ == "__main__":
    main()
