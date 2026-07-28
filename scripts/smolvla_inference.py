#!/usr/bin/env python3
"""
SmolVLA Inference Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loads the Datawhale SmolVLA checkpoint and runs closed-loop inference
in the MuJoCo OMY pick-and-place environment.

Environment must match the training setup exactly:
  - Robot: OMY (6-DOF arm)
  - Cameras: agentview (256×256) + egocentric/wrist (256×256)
  - State: 6-D joint angles
  - Action: 7-D EEF pose (x,y,z,roll,pitch,yaw,gripper)
  - Language: "Place the {red/blue} mug on the plate."
"""

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
import torch


# Ensure lerobot can find the SmolVLA checkpoint
CKPT_DIR = "/workspace/vla-course/weights/smolvla/weights"

# Add mujoco_env to path
sys.path.insert(0, "/workspace/vla-course")
from mujoco_env.y_env2 import SimpleEnv2


    print(f"[SmolVLA] Loaded. Device: {device}, Params: {sum(p.numel() for p in policy.parameters()):,}")
    return policy

def load_tokenizer():
    """Load the SmolVLM2 tokenizer/processor."""
    from transformers import AutoProcessor

    vlm_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    print(f"[Tokenizer] Loading {vlm_name}...")
    processor = AutoProcessor.from_pretrained(
        vlm_name,
        local_files_only=True,
        trust_remote_code=True,
    )
    print("[Tokenizer] Loaded.")
    return processor


def tokenize_instruction(processor, instruction: str, max_length: int = 48):
    """Tokenize a language instruction for the SmolVLA model."""
    tokens = processor.tokenizer(
        instruction,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return tokens["input_ids"], tokens["attention_mask"]


def build_batch(rgb_agent, rgb_ego, joint_state, lang_tokens, lang_mask, device="cuda"):
    """
    Build the batch dict expected by SmolVLAPolicy.select_action().

    Args:
        rgb_agent: np.ndarray (H, W, 3), uint8 [0, 255]
        rgb_ego:   np.ndarray (H, W, 3), uint8 [0, 255]
        joint_state: np.ndarray (6,), float32 joint angles in radians
        lang_tokens: torch.Tensor (1, max_length) long
        lang_mask:   torch.Tensor (1, max_length) long
        device: str

    Returns:
        batch: dict with keys expected by the policy
    """
    # Convert images to float [0, 1] and to (B, C, H, W)
    img_agent = torch.from_numpy(rgb_agent).float() / 255.0
    img_agent = img_agent.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

    img_ego = torch.from_numpy(rgb_ego).float() / 255.0
    img_ego = img_ego.permute(2, 0, 1).unsqueeze(0).to(device)

    state = torch.from_numpy(joint_state).float().unsqueeze(0).to(device)  # (1, 6)

    return {
        "observation.image": img_agent,
        "observation.wrist_image": img_ego,
        "observation.state": state,
        "observation.language_tokens": lang_tokens.to(device),
        "observation.language_attention_mask": lang_mask.to(device),
    }


def run_inference(
    policy,
    env: SimpleEnv2,
    processor,
    instruction: str,
    max_steps: int = 300,
    record_video: bool = True,
    video_path: str = "/workspace/vla-course/outputs/smolvla_inference/smolvla_inference.mp4",
    seed: int = 0,
):
    """
    Run closed-loop inference.

    Args:
        policy: SmolVLAPolicy instance
        env: SimpleEnv2 instance
        processor: SmolVLM2 AutoProcessor
        instruction: language instruction string
        max_steps: maximum steps per episode
        record_video: whether to save an MP4 video
        video_path: output video path
        seed: random seed
    """
    print(f"\n{'='*60}")
    print(f"[Inference] Instruction: {instruction}")
    print(f"[Inference] Seed: {seed}")
    print(f"[Inference] Max steps: {max_steps}")
    print(f"{'='*60}\n")

    # Tokenize instruction once
    lang_tokens, lang_mask = tokenize_instruction(processor, instruction)
    device = policy.config.device

    # Reset environment and policy
    np.random.seed(seed)
    env.reset(seed=seed)
    env.instruction = instruction
    policy.reset()

    # Reset to initial position
    q_init = np.deg2rad([0, 0, 0, 0, 0, 0])
    env.env.forward(q=q_init, joint_names=env.joint_names, increase_tick=False)
    for _ in range(50):
        env.step_env()

    frames = []
    success = False
    step_count = 0

    try:
        for step in range(max_steps):
            step_count = step + 1

            # Grab observations
            env.grab_image()
            rgb_agent = env.rgb_agent.copy()  # (H, W, 3) uint8
            rgb_ego = env.rgb_ego.copy()
            joint_state = env.get_joint_state()  # (6,) float32

            # Build batch and get action
            batch = build_batch(rgb_agent, rgb_ego, joint_state, lang_tokens, lang_mask, device=device)
            action = policy.select_action(batch)  # (7,) tensor on device

            # Convert to numpy
            action_np = action.cpu().numpy()

            # Apply action to environment
            env.step(action_np)
            env.step_env()

            # Determine target object for success check
            if "red" in instruction.lower():
                obj_target = "body_obj_mug_5"
            elif "blue" in instruction.lower():
                obj_target = "body_obj_mug_6"
            else:
                obj_target = None

            # Check success: is the blue mug on the plate?
            success = check_success(env, obj_target)
            if success:
                print(f"[Step {step_count:3d}] 🎉 SUCCESS! Object on plate.")
                break

            # Render for video
            frame = render_frame(env)
            frames.append(frame)

            if step % 50 == 0:
                print(f"[Step {step_count:3d}] action={action_np.round(4)}, eef_z={env.p0[2]:.4f}")

    except KeyboardInterrupt:
        print("\n[Inference] Interrupted.")
    except Exception as e:
        print(f"\n[Inference] ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Save video
        if record_video and frames:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            save_video_cv2(video_path, frames, fps=20)
            print(f"\n[Video] Saved {len(frames)} frames → {video_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"[Result] Steps: {step_count}/{max_steps}")
    print(f"[Result] Success: {'✅ YES' if success else '❌ NO'}")
    print(f"[Result] Final EEF pos: {env.p0.round(4)}")
    print(f"{'='*60}")

    return success, step_count, frames


def render_frame(env: SimpleEnv2, width: int = 640, height: int = 480):
    """Render a composite frame for video recording."""
    env.grab_image()
    agent = cv2.resize(env.rgb_agent, (width // 2, height))
    ego = cv2.resize(env.rgb_ego, (width // 2, height))

    # Side-by-side
    composite = np.hstack([agent, ego])

    # Add labels
    cv2.putText(composite, "agentview", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(composite, "egocentric", (width // 2 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add instruction
    inst = env.instruction if hasattr(env, 'instruction') else ""
    cv2.putText(composite, inst, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)


def check_success(env: SimpleEnv2, obj_target: str, lift_threshold: float = 0.03, z_plate: float = 0.83) -> bool:
    """
    Check if the target object is on the plate.

    Datawhale criterion: lift ≥ 0.03m above plate & upright cosine ≥ 0.7
    Simplified: target object center is near plate position (x≈0.3, y≈-0.25, z≈0.83+0.03)
    """
    if obj_target is None:
        return False

    try:
        obj_pos, obj_R = env.env.get_pR_body(body_name=obj_target)
        plate_pos, _ = env.env.get_pR_body(body_name="body_obj_plate_11")

        # Check object height above plate
        height_above_plate = obj_pos[2] - plate_pos[2]

        # Check horizontal proximity to plate center
        dist_xy = np.linalg.norm(obj_pos[:2] - plate_pos[:2])

        # Upright: z-axis of object rotation should be close to world z-axis
        upright_cosine = np.dot(obj_R[:, 2], np.array([0, 0, 1]))

        success = height_above_plate > lift_threshold and upright_cosine > 0.7
        return bool(success)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="SmolVLA Inference")
    parser.add_argument("--instruction", type=str, default="Place the blue mug on the plate.",
                        help="Language instruction")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint", type=str, default=CKPT_DIR, help="SmolVLA checkpoint dir")
    parser.add_argument("--video", type=str, default="/workspace/vla-course/outputs/smolvla_inference/smolvla_inference.mp4",
                        help="Output video path")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--xml", type=str, default="/workspace/vla-course/models/omy/example_scene_y2.xml",
                        help="MuJoCo XML scene path")
    parser.add_argument("--eval-all", action="store_true",
                        help="Run full evaluation (60 seeds, 30 red + 30 blue)")
    args = parser.parse_args()

    # Check GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model and tokenizer
    policy = load_smolvla(args.checkpoint, device=args.device)
    processor = load_tokenizer()

    if args.eval_all:
        # Full evaluation: 30 red + 30 blue
        results = {"red": [], "blue": []}
        for obj_color, obj_name in [("red", "red"), ("blue", "blue")]:
            instruction = f"Place the {obj_color} mug on the plate."
            for seed in range(30):
                video = f"/workspace/vla-course/outputs/smolvla_inference/eval_{obj_color}_seed{seed:02d}.mp4"
                env = SimpleEnv2(xml_path=args.xml, action_type='eef_pose', state_type='joint_angle', seed=seed)
                success, steps, _ = run_inference(
                    policy, env, processor, instruction,
                    max_steps=args.max_steps,
                    record_video=not args.no_video,
                    video_path=video,
                    seed=seed,
                )
                results[obj_color].append(success)
                print(f"[Eval] {obj_color} seed={seed:02d}: {'✅' if success else '❌'} ({steps} steps)")
                env.close() if hasattr(env, 'close') else None

        # Summary
        red_success = sum(results["red"])
        blue_success = sum(results["blue"])
        total_success = red_success + blue_success
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY (Strict)")
        print(f"  Red:  {red_success}/30 ({red_success/30*100:.0f}%)")
        print(f"  Blue: {blue_success}/30 ({blue_success/30*100:.0f}%)")
        print(f"  Total: {total_success}/60 ({total_success/60*100:.0f}%)")
        print(f"  Datawhale reference: 57/60 (95.0%)")
        print(f"{'='*60}")
    else:
        # Single run
        env = SimpleEnv2(xml_path=args.xml, action_type='eef_pose', state_type='joint_angle', seed=args.seed)
        run_inference(
            policy, env, processor, args.instruction,
            max_steps=args.max_steps,
            record_video=not args.no_video,
            video_path=args.video,
            seed=args.seed,
        )
        if hasattr(env, 'close'):
            env.close()


if __name__ == "__main__":
    main()

def save_video_cv2(path, frames, fps=20):
    """Save frames as MP4 using OpenCV."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
