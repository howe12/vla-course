import os, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_STATE, ACTION

device = torch.device("cuda")
state_dim, action_dim = 6, 6
img_key = "observation.images.top"
config = ACTConfig(
    input_features={
        img_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    },
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    pretrained_backbone_weights=None,
    use_vae=True, chunk_size=16, n_action_steps=16,
)
policy = ACTPolicy(config).to(device).eval()

B = 1
obs = {
    img_key: torch.rand(B, 3, 480, 640, device=device),
    OBS_STATE: torch.rand(B, state_dim, device=device),
}

# warmup
with torch.no_grad():
    for _ in range(5):
        policy.reset()
        policy.select_action(obs)
torch.cuda.synchronize()

# benchmark
N = 50
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N):
        policy.reset()
        a = policy.select_action(obs)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N
print(f"=== ACT inference latency (batch=1, 480x640, 25W mode) ===")
print(f"per-call latency: {dt*1000:.1f} ms")
print(f"throughput: {1/dt:.1f} actions/s")
print(f"GPU: {torch.cuda.get_device_name(0)}")
