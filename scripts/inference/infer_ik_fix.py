#!/usr/bin/env python3
"""IK eps 1e-4 inference — dataset joints + object ranges."""
import os, sys, json, cv2, numpy as np, torch, time

sys.path.insert(0, "/workspace/vla-course")
from mujoco_env.y_env2 import SimpleEnv2

DEVICE = "cuda"
CKPT = "/workspace/vla-course/weights/smolvla_datawhale/weights"
DATASET_ROOT = "/tmp/demo_data_language"
XML = "/workspace/vla-course/models/omy/example_scene_y2.xml"
OUT_DIR = "/workspace/vla-course/outputs/course_capture/ik_1e-4"
INSTRUCTION = "Place the blue mug on the plate."
MAX_ENV_STEPS = 300
SEED = 0
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_INIT_JOINTS = np.array([0.385, -0.111, 1.177, 0.511, 1.570, -0.385], dtype=np.float32)
DATASET_OBJ_RANGES = {
    "mug_red_x":  [0.320, 0.330], "mug_red_y":  [0.001, 0.019], "mug_red_z":  0.83,
    "mug_blue_x": [0.290, 0.299], "mug_blue_y": [0.190, 0.210], "mug_blue_z": 0.83,
    "plate_x": 0.30, "plate_y": -0.25, "plate_z": 0.82,
}

# ── Load policy ──
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features

ds_meta = LeRobotDatasetMetadata("datawhale_eai_pnp_language", root=DATASET_ROOT)
features = dataset_to_policy_features(ds_meta.features)
out_feats = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
in_feats  = {k: v for k, v in features.items() if k not in out_feats}
cfg = SmolVLAConfig(input_features=in_feats, output_features=out_feats, chunk_size=50, n_action_steps=50)
policy = SmolVLAPolicy.from_pretrained(CKPT, config=cfg, dataset_stats=ds_meta.stats)
policy.to(DEVICE); policy.eval(); policy.reset()

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True, trust_remote_code=True)
tokens = processor.tokenizer(INSTRUCTION, return_tensors="pt", padding="max_length", truncation=True, max_length=48)

# ── Layer 0: Init + verify ──
np.random.seed(SEED)
env = SimpleEnv2(xml_path=XML, action_type="eef_pose", state_type="joint_angle", seed=SEED)
env.reset(seed=SEED)

q_init = DATASET_INIT_JOINTS.copy()
env.env.forward(q=q_init, joint_names=env.joint_names, increase_tick=False)
env.last_q = q_init.copy()
env.q = np.concatenate([q_init, np.array([0.0, 0.0, 0.0, 0.0])])
env.p0, env.R0 = env.env.get_pR_body(body_name="tcp_link")
print(f"[INIT] Joints: {DATASET_INIT_JOINTS}")
print(f"[INIT] EEF pos: {env.p0} (z={env.p0[2]:.4f})")

# Place objects
mug_red_p = np.array([np.random.uniform(*DATASET_OBJ_RANGES["mug_red_x"]), np.random.uniform(*DATASET_OBJ_RANGES["mug_red_y"]), DATASET_OBJ_RANGES["mug_red_z"]])
mug_blue_p = np.array([np.random.uniform(*DATASET_OBJ_RANGES["mug_blue_x"]), np.random.uniform(*DATASET_OBJ_RANGES["mug_blue_y"]), DATASET_OBJ_RANGES["mug_blue_z"]])
plate_p = np.array([DATASET_OBJ_RANGES["plate_x"], DATASET_OBJ_RANGES["plate_y"], DATASET_OBJ_RANGES["plate_z"]])
env.env.set_p_base_body("body_obj_mug_5", p=mug_red_p); env.env.set_R_base_body("body_obj_mug_5", R=np.eye(3))
env.env.set_p_base_body("body_obj_mug_6", p=mug_blue_p); env.env.set_R_base_body("body_obj_mug_6", R=np.eye(3))
env.env.set_p_base_body("body_obj_plate_11", p=plate_p); env.env.set_R_base_body("body_obj_plate_11", R=np.eye(3))
env.env.forward(increase_tick=False)
env.obj_target = "body_obj_mug_6"
for _ in range(50): env.step_env()

# ── Layer 0: Capture PRE-LOOP initial frame ──
main_img_init, wrist_img_init = env.grab_image()
cv2.imwrite(f"{OUT_DIR}/init_main.jpg", cv2.cvtColor(main_img_init, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{OUT_DIR}/init_wrist.jpg", cv2.cvtColor(wrist_img_init, cv2.COLOR_RGB2BGR))
print(f"[LAYER0] Initial frames saved. EEF z={env.p0[2]:.4f}")

# ── Inference ──
def pad_img(img, target=512):
    h, w, _ = img.shape
    pad_h = max(h, w) - h; pad_w = max(h, w) - w
    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    return cv2.resize(img_pad, (target, target)).astype(np.float32) / 255.0

gap_hist, mugz_hist, grip_hist, anoms = [], [], [], []
frames = []
t_start = time.time()
model_calls = 0

sep = "=" * 60
print(f"\n{sep}")
print(f"[INFERENCE] IK eps=1e-4 | dataset joints | dataset obj ranges")
print(f"{sep}\n")

for step in range(MAX_ENV_STEPS):
    t0 = time.time()
    state = env.get_joint_state()[:6]
    image, wrist_image = env.grab_image()

    img_tensor = torch.from_numpy(pad_img(image)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    wrist_tensor = torch.from_numpy(pad_img(wrist_image)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

    obs = {
        "observation.image": img_tensor,
        "observation.wrist_image": wrist_tensor,
        "observation.state": state_tensor,
        "observation.language.tokens": tokens["input_ids"].to(DEVICE),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(DEVICE),
    }

    q_before = len(policy._queues["action"])
    with torch.inference_mode():
        action_tensor = policy.select_action(obs)
    if len(policy._queues["action"]) > q_before:
        model_calls += 1

    action = action_tensor.squeeze().cpu().numpy()

    # Layer 1: anomaly checks
    if abs(action[0]) > 0.1 or abs(action[1]) > 0.1:
        anoms.append(f"s{step}: dx={action[0]:.4f} dy={action[1]:.4f} OOD")
    if abs(action[6]) > 1.5:
        anoms.append(f"s{step}: grip={action[6]:.3f} OOD")

    env.step(action)
    env.step_env()

    p_red, p_blue, p_plate = env.get_obj_pose()
    mug_z, gap = p_blue[2], p_blue[2] - p_plate[2]
    gap_hist.append(gap); mugz_hist.append(mug_z)
    grip_hist.append(float(action[6]))

    if step % 100 == 0:
        main_img, _ = env.grab_image()
        frames.append(main_img)
        print(f"Step {step:3d} | mug_z={mug_z:.4f} gap={gap:+.4f} "
              f"eef_z~{env.p0[2]:.3f} grip={action[6]:.3f} calls={model_calls}")

t_total = time.time() - t_start

# ── Layer 2: Analysis ──
video_path = f"{OUT_DIR}/ik1e4_600.mp4"
if frames:
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for f in frames: writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

avg_grip = round(float(np.mean(grip_hist)), 3)

print(f"\n{sep}")
print(f"[SUMMARY] Time: {t_total:.1f}s  Calls: {model_calls}")
print(f"[SUMMARY] Mug z: {min(mugz_hist):.4f} -> {max(mugz_hist):.4f} -> {mugz_hist[-1]:.4f}")
print(f"[SUMMARY] Gap: min={min(gap_hist):+.4f} final={gap_hist[-1]:+.4f}")
print(f"[SUMMARY] Avg grip: {avg_grip}  Anomalies: {len(anoms)}")
if anoms[:5]: print(f"[ANOM] First 5: {anoms[:5]}")
print(f"[SAVE] Video: {video_path}")
print(f"{sep}")
