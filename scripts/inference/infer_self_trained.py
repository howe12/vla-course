#!/usr/bin/env python3
"""Inference with self-trained SmolVLA — select_action closed-loop (queue-based)."""
import os, sys, json, cv2, numpy as np, torch, time

sys.path.insert(0, "/workspace/vla-course")
from mujoco_env.y_env2 import SimpleEnv2

DEVICE = "cuda"
CKPT = "/workspace/vla-course/outputs/smolvla_v1/checkpoints/005000/pretrained_model"
DATASET_ROOT = "/tmp/demo_data_language"
XML = "/workspace/vla-course/models/omy/example_scene_y2.xml"
OUT_DIR = "/workspace/vla-course/outputs/course_capture/self_trained"
INSTRUCTION = "Place the blue mug on the plate."
MAX_ENV_STEPS = 600
SEED = 0

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load policy ──────────────────────────────────────────────
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features

print(f"[LOAD] Dataset metadata from {DATASET_ROOT}")
ds_meta = LeRobotDatasetMetadata("datawhale_eai_pnp_language", root=DATASET_ROOT)

features = dataset_to_policy_features(ds_meta.features)
out_feats = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
in_feats  = {k: v for k, v in features.items() if k not in out_feats}

cfg = SmolVLAConfig(input_features=in_feats, output_features=out_feats,
                     chunk_size=50, n_action_steps=50)

print(f"[LOAD] Policy from {CKPT}")
policy = SmolVLAPolicy.from_pretrained(CKPT, config=cfg, dataset_stats=ds_meta.stats)
policy.to(DEVICE)
policy.eval()
policy.reset()
print(f"[LOAD] {sum(p.numel() for p in policy.parameters()):,} params")

# ── Language tokens ──────────────────────────────────────────
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    local_files_only=True, trust_remote_code=True
)
tokens = processor.tokenizer(
    INSTRUCTION, return_tensors="pt",
    padding="max_length", truncation=True, max_length=48
)
lang_tokens = tokens["input_ids"]
lang_mask = tokens["attention_mask"].bool()

# ── Env ──────────────────────────────────────────────────────
env = SimpleEnv2(xml_path=XML, action_type='eef_pose', state_type="joint_angle", seed=SEED)
env.instruction = INSTRUCTION
env.reset(seed=SEED)
q_init = np.deg2rad([0, 0, 0, 0, 0, 0])
env.env.forward(q=q_init, joint_names=env.joint_names, increase_tick=False)
for _ in range(50):
    env.step_env()

# ── Metrics ──────────────────────────────────────────────────
gap_history, mug_z_history, gripper_history, step_times = [], [], [], []
model_call_count = 0

def pad_img(img, target=512):
    h, w, _ = img.shape
    max_dim = max(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w
    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    img_resized = cv2.resize(img_pad, (target, target))
    return img_resized.astype(np.float32) / 255.0

print(f"\n{'='*60}")
print(f"[SELF-TRAINED] {INSTRUCTION}")
print(f"[SELF-TRAINED] select_action + n_action_steps=50, closed-loop")
print(f"{'='*60}\n")

frames = []
t_start = time.time()

for step in range(MAX_ENV_STEPS):
    t0 = time.time()

    # Get state and images
    state = env.get_joint_state()[:6]
    image, wrist_image = env.grab_image()

    img_tensor = torch.from_numpy(pad_img(image)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    wrist_tensor = torch.from_numpy(pad_img(wrist_image)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

    obs = {
        "observation.image": img_tensor,
        "observation.wrist_image": wrist_tensor,
        "observation.state": state_tensor,
        "observation.language.tokens": lang_tokens.to(DEVICE),
        "observation.language.attention_mask": lang_mask.to(DEVICE),
    }

    # Track when model is actually called
    queue_len_before = len(policy._queues["action"])

    with torch.inference_mode():
        action_tensor = policy.select_action(obs)

    queue_len_after = len(policy._queues["action"])
    if queue_len_after > queue_len_before:
        model_call_count += 1

    action = action_tensor.squeeze().cpu().numpy()
    env.step(action)
    env.step_env()

    # Record metrics
    p_mug_red, p_mug_blue, p_plate = env.get_obj_pose()
    mug_z = p_mug_blue[2]
    plate_z = p_plate[2]
    gap = mug_z - plate_z

    gap_history.append(gap)
    mug_z_history.append(mug_z)
    gripper_history.append(float(action[6]))

    t1 = time.time()
    step_times.append(t1 - t0)

    # Render frame every 100 env steps (main camera from grab_image)
    if step % 100 == 0:
        main_img, _ = env.grab_image()
        frames.append(main_img)

    if step % 100 == 0:
        q = env.env.get_qpos_joints(joint_names=env.joint_names)
        print(f"Step {step:5d} | mug_z={mug_z:.4f} gap={gap:.4f} "
              f"grip={action[6]:.3f} model_calls={model_call_count} "
              f"q={np.round(q[:6], 2)}")

t_total = time.time() - t_start

# ── Save video ───────────────────────────────────────────────
video_path = f"{OUT_DIR}/self_trained_600.mp4"
if frames:
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.shape[-1] == 3 else f)
    writer.release()
    print(f"\n[SAVE] Video: {video_path} ({len(frames)} frames)")

# ── Save metrics ─────────────────────────────────────────────
summary = {
    "total_time_s": round(t_total, 1),
    "env_steps": MAX_ENV_STEPS,
    "model_calls": model_call_count,
    "max_mug_z": round(max(mug_z_history), 4),
    "min_gap": round(min(gap_history), 4),
    "final_gap": round(gap_history[-1], 4),
    "final_mug_z": round(mug_z_history[-1], 4),
    "avg_step_ms": round(t_total / MAX_ENV_STEPS * 1000, 1),
    "avg_gripper": round(float(np.mean(gripper_history)), 3),
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"[SUMMARY] Total: {t_total:.1f}s  Model calls: {model_call_count}")
print(f"[SUMMARY] Max mug_z: {summary['max_mug_z']}m")
print(f"[SUMMARY] Min gap:   {summary['min_gap']}m")
print(f"[SUMMARY] Final gap: {summary['final_gap']}m")
print(f"[SUMMARY] Avg gripper: {summary['avg_gripper']}")
print(f"{'='*60}")
