import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_STATE, ACTION

print("=== device ===")
assert torch.cuda.is_available(), "CUDA not available!"
device = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0))

print("=== build ACT config (from scratch, no pretrained backbone) ===")
state_dim = 6
action_dim = 6
img_key = "observation.images.top"
config = ACTConfig(
    input_features={
        img_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    },
    output_features={
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    },
    pretrained_backbone_weights=None,  # offline: random ResNet init
    use_vae=True,
    chunk_size=16,
    n_action_steps=16,
)

print("=== instantiate ACTPolicy ===")
policy = ACTPolicy(config)
policy = policy.to(device)
policy.train()
n_params = sum(p.numel() for p in policy.parameters())
print(f"params: {n_params/1e6:.1f}M")

print("=== build dummy batch on GPU ===")
B = 2
batch = {
    img_key: torch.rand(B, 3, 480, 640, device=device),
    OBS_STATE: torch.rand(B, state_dim, device=device),
    ACTION: torch.rand(B, config.chunk_size, action_dim, device=device),
    "action_is_pad": torch.zeros(B, config.chunk_size, dtype=torch.bool, device=device),
}

print("=== forward pass on GPU (training mode, VAE) ===")
loss, loss_dict = policy.forward(batch)
print("loss:", float(loss))
print("loss_dict:", loss_dict)
assert torch.isfinite(loss), "loss is not finite!"

print("=== inference: select_action on GPU ===")
policy.eval()
policy.reset()
obs_batch = {
    img_key: torch.rand(B, 3, 480, 640, device=device),
    OBS_STATE: torch.rand(B, state_dim, device=device),
}
with torch.no_grad():
    action = policy.select_action(obs_batch)
print("action shape:", tuple(action.shape), "device:", action.device)
assert action.device.type == "cuda", "action not on GPU!"
print("=== ACT GPU SMOKE TEST PASSED ===")
