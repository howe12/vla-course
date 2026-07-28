#!/usr/bin/env python3
"""
Inference with official weights — using ONLY dataset-verified values.
No guessing, no IK, no custom overrides.

Initial joint angles:  from dataset observation.state (all 20 episodes identical)
Object positions:     from dataset obj_init (actual collection distribution)
"""
import os, sys, json, cv2, numpy as np, torch, time, copy

sys.path.insert(0, "/workspace/vla-course")
from mujoco_env.y_env2 import SimpleEnv2
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r

DEVICE = "cuda"
CKPT = "/workspace/vla-course/weights/smolvla_datawhale/weights"
DATASET_ROOT = "/tmp/demo_data_language"
XML = "/workspace/vla-course/models/omy/example_scene_y2.xml"
OUT_DIR = "/workspace/vla-course/outputs/course_capture/dataset_match"
INSTRUCTION = "Place the blue mug on the plate."
MAX_ENV_STEPS = 600
SEED = 0

os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# VALUES FROM DATASET — not guessed, not inferred
# ═══════════════════════════════════════════════════════════════

# Initial joint angles: extracted from observation.state of all 20 episodes
# Source: /tmp/demo_data_language, datawhale_eai_pnp_language
# All 20 episodes start from identical joint config (std ≈ 0.000)
DATASET_INIT_JOINTS = np.array([0.385, -0.111, 1.177, 0.511, 1.570, -0.385],
                                dtype=np.float32)

# Object positions: extracted from obj_init of all 20 episodes
# Source: /tmp/demo_data_language, datawhale_eai_pnp_language
# These are TIGHT ranges — NOT the wide sampling in y_env_dw.py
DATASET_OBJ_RANGES = {
    "mug_red_x":  [0.320, 0.330],  # red mug (body_obj_mug_5)
    "mug_red_y":  [0.001, 0.019],
    "mug_red_z":  0.83,
    "mug_blue_x": [0.290, 0.299],  # blue mug (body_obj_mug_6) — TARGET
    "mug_blue_y": [0.190, 0.210],
    "mug_blue_z": 0.83,
    "plate_x":    0.30,            # plate (body_obj_plate_11)
    "plate_y":   -0.25,
    "plate_z":    0.82,
}

print("=" * 60)
print("DATASET-MATCH INFERENCE (no guessing)")
print(f"Robot init joints: {DATASET_INIT_JOINTS}")
print(f"Red mug  x:[{DATASET_OBJ_RANGES['mug_red_x'][0]:.3f},{DATASET_OBJ_RANGES['mug_red_x'][1]:.3f}]")
print(f"         y:[{DATASET_OBJ_RANGES['mug_red_y'][0]:.3f},{DATASET_OBJ_RANGES['mug_red_y'][1]:.3f}]")
print(f"Blue mug x:[{DATASET_OBJ_RANGES['mug_blue_x'][0]:.3f},{DATASET_OBJ_RANGES['mug_blue_x'][1]:.3f}]")
print(f"         y:[{DATASET_OBJ_RANGES['mug_blue_y'][0]:.3f},{DATASET_OBJ_RANGES['mug_blue_y'][1]:.3f}]")
print(f"Plate    : [{DATASET_OBJ_RANGES['plate_x']}, {DATASET_OBJ_RANGES['plate_y']}, {DATASET_OBJ_RANGES['plate_z']}]")
print("=" * 60)

# ── Load policy (official weights) ───────────────────────────
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features

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

# ── Language tokens ──────────────────────────────────────────
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    local_files_only=True, trust_remote_code=True
)
tokens = processor.tokenizer(INSTRUCTION, return_tensors="pt",
                              padding="max_length", truncation=True, max_length=48)

# ── Environment — use dataset values directly ─────────────────
np.random.seed(SEED)
env = SimpleEnv2(xml_path=XML, action_type='eef_pose', state_type="joint_angle", seed=SEED)
env.instruction = INSTRUCTION

# Standard reset first (needed for MuJoCo init)
env.reset(seed=SEED)

# ── Set robot initial joints DIRECTLY (from dataset, no IK guesswork) ──
q_init = DATASET_INIT_JOINTS.copy()
env.env.forward(q=q_init, joint_names=env.joint_names, increase_tick=False)
env.last_q = q_init.copy()
env.q = np.concatenate([q_init, np.array([0.0, 0.0, 0.0, 0.0])])  # +4 gripper joints
env.p0, env.R0 = env.env.get_pR_body(body_name='tcp_link')
print(f"[ROBOT] Joint angles set to dataset values")
print(f"[ROBOT] EEF pos after forward: {env.p0}")

# ── Set object positions from dataset ranges ──────────────────
mug_red_p = np.array([
    np.random.uniform(*DATASET_OBJ_RANGES["mug_red_x"]),
    np.random.uniform(*DATASET_OBJ_RANGES["mug_red_y"]),
    DATASET_OBJ_RANGES["mug_red_z"],
])
mug_blue_p = np.array([
    np.random.uniform(*DATASET_OBJ_RANGES["mug_blue_x"]),
    np.random.uniform(*DATASET_OBJ_RANGES["mug_blue_y"]),
    DATASET_OBJ_RANGES["mug_blue_z"],
])
plate_p = np.array([
    DATASET_OBJ_RANGES["plate_x"],
    DATASET_OBJ_RANGES["plate_y"],
    DATASET_OBJ_RANGES["plate_z"],
])

env.env.set_p_base_body(body_name='body_obj_mug_5', p=mug_red_p)
env.env.set_R_base_body(body_name='body_obj_mug_5', R=np.eye(3, 3))
env.env.set_p_base_body(body_name='body_obj_mug_6', p=mug_blue_p)
env.env.set_R_base_body(body_name='body_obj_mug_6', R=np.eye(3, 3))
env.env.set_p_base_body(body_name='body_obj_plate_11', p=plate_p)
env.env.set_R_base_body(body_name='body_obj_plate_11', R=np.eye(3, 3))
env.env.forward(increase_tick=False)

env.obj_target = 'body_obj_mug_6'  # blue mug is the target
print(f"[OBJECTS] Red:  {mug_red_p}")
print(f"[OBJECTS] Blue: {mug_blue_p} (target)")
print(f"[OBJECTS] Plate: {plate_p}")

for _ in range(50):
    env.step_env()

# ── Inference loop ────────────────────────────────────────────
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
print(f"[INFERENCE] Official weights | dataset initial joints | dataset object ranges")
print(f"{'='*60}\n")

frames = []
t_start = time.time()

for step in range(MAX_ENV_STEPS):
    t0 = time.time()

    state = env.get_joint_state()[:6]
    image, wrist_image = env.grab_image()

    img_tensor = torch.from_numpy(pad_img(image)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    wrist_tensor = torch.from_numpy(pad_img(wrist_image)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

    obs = {
        "observation.image": img_tensor,
        "observation.wrist_image": wrist_tensor,
        "observation.state": state_tensor,
        "observation.language.tokens": tokens["input_ids"].to(DEVICE),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(DEVICE),
    }

    queue_len_before = len(policy._queues["action"])
    with torch.inference_mode():
        action_tensor = policy.select_action(obs)
    queue_len_after = len(policy._queues["action"])
    if queue_len_after > queue_len_before:
        model_call_count += 1

    action = action_tensor.squeeze().cpu().numpy()
    env.step(action)
    env.step_env()

    p_red, p_blue, p_plate = env.get_obj_pose()
    mug_z = p_blue[2]
    gap = mug_z - p_plate[2]

    gap_history.append(gap)
    mug_z_history.append(mug_z)
    gripper_history.append(float(action[6]))

    step_times.append(time.time() - t0)

    if step % 100 == 0:
        main_img, _ = env.grab_image()
        frames.append(main_img)
        print(f"Step {step:3d} | mug_z={mug_z:.4f} gap={gap:+.4f} "
              f"mug_xy=({p_blue[0]:.3f},{p_blue[1]:.3f}) "
              f"grip={action[6]:.3f} calls={model_call_count}")

t_total = time.time() - t_start

# ── Save ─────────────────────────────────────────────────────
video_path = f"{OUT_DIR}/dataset_match_600.mp4"
if frames:
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"\n[SAVE] Video: {video_path} ({len(frames)} frames, {os.path.getsize(video_path)//1024}KB)")

summary = {
    "approach": "dataset_verified_only",
    "init_joints": DATASET_INIT_JOINTS.tolist(),
    "objects": {"red": mug_red_p.tolist(), "blue": mug_blue_p.tolist(), "plate": plate_p.tolist()},
    "total_time_s": round(t_total, 1),
    "env_steps": MAX_ENV_STEPS,
    "model_calls": model_call_count,
    "max_mug_z": round(max(mug_z_history), 4),
    "min_gap": round(min(gap_history), 4),
    "final_gap": round(gap_history[-1], 4),
    "avg_gripper": round(float(np.mean(gripper_history)), 3),
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"[SUMMARY] Total: {t_total:.1f}s  Model calls: {model_call_count}")
print(f"[SUMMARY] Max mug_z: {summary['max_mug_z']}m  Min gap: {summary['min_gap']}m")
print(f"[SUMMARY] Final gap: {summary['final_gap']}m  Avg gripper: {summary['avg_gripper']}")
print(f"{'='*60}")
