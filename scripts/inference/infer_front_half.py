#!/usr/bin/env python3
"""Inference with official weights + Datawhale-style object placement (front half)."""
import os, sys, json, cv2, numpy as np, torch, time, copy

sys.path.insert(0, "/workspace/vla-course")
from mujoco_env.y_env2 import SimpleEnv2
from mujoco_env.utils import sample_xyzs

DEVICE = "cuda"
CKPT = "/workspace/vla-course/weights/smolvla_datawhale/weights"
DATASET_ROOT = "/tmp/demo_data_language"
XML = "/workspace/vla-course/models/omy/example_scene_y2.xml"
OUT_DIR = "/workspace/vla-course/outputs/course_capture/front_half"
INSTRUCTION = "Place the blue mug on the plate."
MAX_ENV_STEPS = 600
SEED = 0

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load policy (official weights) ───────────────────────────
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

# ── Env with Datawhale object placement ──────────────────────
np.random.seed(SEED)

env = SimpleEnv2(xml_path=XML, action_type='eef_pose', state_type="joint_angle", seed=SEED)
env.instruction = INSTRUCTION

# Standard reset (sets robot home with adjusted view)
env.reset(seed=SEED)
# Override robot init: pull back and higher so wrist camera sees front-half objects
# Datawhade objects are at x=[0.24,0.40], z=0.82 — camera at [0.2,0.0,1.05] gives ~23cm clearance
q_init = np.deg2rad([0, 0, 0, 0, 0, 0])
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r
init_p = np.array([0.2, 0.0, 1.05], dtype=np.float32)
init_rpy = np.deg2rad([90, 0, 90])
q_zero, _, _ = solve_ik(
    env=env.env,
    joint_names_for_ik=env.joint_names,
    body_name_trgt='tcp_link',
    q_init=q_init,
    p_trgt=init_p,
    R_trgt=rpy2r(init_rpy),
)
env.env.forward(q=q_zero, joint_names=env.joint_names, increase_tick=False)
env.last_q = q_zero.copy()
env.q = np.concatenate([q_zero, np.array([0.0]*4)])
env.p0, env.R0 = env.env.get_pR_body(body_name='tcp_link')
print(f"  Robot EEF init: p={init_p}, rpy_deg=[90,0,90]")

# ═══ OVERRIDE: Datawhale-style object placement (front half) ═══
# Datawhale uses: x_range=[0.24, 0.40], y_range=[-0.20, 0.20], min_dist=0.20
# Objects: mug_5 (red), mug_6 (blue), plate_11
obj_names = ['body_obj_mug_5', 'body_obj_mug_6', 'body_obj_plate_11']
obj_xyzs = sample_xyzs(
    n_sample=3,
    x_range=[+0.24, +0.40],
    y_range=[-0.20, +0.20],
    z_range=[0.82, 0.82],
    min_dist=0.20,
    xy_margin=0.0
)
for idx, name in enumerate(obj_names):
    env.env.set_p_base_body(body_name=name, p=obj_xyzs[idx, :])
    env.env.set_R_base_body(body_name=name, R=np.eye(3, 3))

# Re-init p0/R0 after object placement
env.env.forward(increase_tick=False)
env.p0, env.R0 = env.env.get_pR_body(body_name='tcp_link')

# Set target
env.obj_target = 'body_obj_mug_6'  # blue mug

print(f"  Objects placed:")
for i, name in enumerate(obj_names):
    print(f"    {name}: {obj_xyzs[i]}")

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
print(f"[FRONT-HALF] {INSTRUCTION}")
print(f"[FRONT-HALF] Official weights + Datawhale object ranges")
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
        "observation.language.tokens": lang_tokens.to(DEVICE),
        "observation.language.attention_mask": lang_mask.to(DEVICE),
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

    p_mug_red, p_mug_blue, p_plate = env.get_obj_pose()
    mug_z = p_mug_blue[2]
    plate_z = p_plate[2]
    gap = mug_z - plate_z

    gap_history.append(gap)
    mug_z_history.append(mug_z)
    gripper_history.append(float(action[6]))

    t1 = time.time()
    step_times.append(t1 - t0)

    if step % 100 == 0:
        main_img, _ = env.grab_image()
        frames.append(main_img)

    if step % 100 == 0:
        p_blue = p_mug_blue
        print(f"Step {step:3d} | mug_z={mug_z:.4f} gap={gap:+.4f} "
              f"mug_xy=({p_blue[0]:.3f},{p_blue[1]:.3f}) "
              f"grip={action[6]:.3f} calls={model_call_count}")

t_total = time.time() - t_start

# ── Save ─────────────────────────────────────────────────────
video_path = f"{OUT_DIR}/front_half_600.mp4"
if frames:
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"\n[SAVE] Video: {video_path} ({len(frames)} frames)")

summary = {
    "config": "official_weights + datawhale_front_half",
    "total_time_s": round(t_total, 1),
    "env_steps": MAX_ENV_STEPS,
    "model_calls": model_call_count,
    "max_mug_z": round(max(mug_z_history), 4),
    "min_gap": round(min(gap_history), 4),
    "final_gap": round(gap_history[-1], 4),
    "final_mug_z": round(mug_z_history[-1], 4),
    "avg_gripper": round(float(np.mean(gripper_history)), 3),
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"[SUMMARY] Total: {t_total:.1f}s  Model calls: {model_call_count}")
print(f"[SUMMARY] Max mug_z: {summary['max_mug_z']}m")
print(f"[SUMMARY] Min gap:   {summary['min_gap']}m")
print(f"[SUMMARY] Final gap: {summary['final_gap']}m")
print(f"{'='*60}")
