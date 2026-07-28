# VLA Course — AMD ROCm Deployment

## One-Click Setup

```bash
git clone https://github.com/howe12/vla-course.git
cd vla-course && git checkout amd-rocm-backup
bash setup.sh
```

## Options

| Flag | Description |
|------|-------------|
| `--skip-rocm` | Skip ROCm driver install |
| `--skip-weights` | Skip model weight download |

## Run Inference

```bash
source venv/bin/activate
python scripts/inference/infer_ik_fix.py
```

## Key Files

| File | Description |
|------|-------------|
| `mujoco_env/ik.py` | IK solver, eps=1e-3 |
| `mujoco_env/y_env2.py` | Environment (eef_pose mode) |
| `scripts/inference/infer_ik_fix.py` | Inference with Layer 0-2 validation |
| `configs/smolvla_omy.yaml` | SmolVLA training config |

## Lessons Learned

1. IK eps must not exceed 1e-3 (container overload at 1e-4)
2. Initialize joints from dataset, NOT IK guess
3. Start Xvfb before inference (DISPLAY=:99)
4. Capture initial frame before inference loop (Layer 0 check)
5. Use tini for process management to prevent zombie processes
