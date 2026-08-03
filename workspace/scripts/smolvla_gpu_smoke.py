import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION

print("=== device ===")
assert torch.cuda.is_available(), "CUDA not available!"
device = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0))

print("=== build config (from scratch, no pretrained weights) ===")
state_dim = 6
action_dim = 6
img_key = "observation.images.top"
config = SmolVLAConfig(
    input_features={
        img_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    },
    output_features={
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    },
    load_vlm_weights=False,   # init from scratch -> no weight files needed
    freeze_vision_encoder=False,
    chunk_size=8,
    n_action_steps=8,
    num_steps=4,
)
print("vlm_model_name:", config.vlm_model_name)

print("=== instantiate policy ===")
policy = SmolVLAPolicy(config)
policy = policy.to(device)
policy.eval()
n_params = sum(p.numel() for p in policy.parameters())
print(f"params: {n_params/1e6:.1f}M")

print("=== build dummy batch on GPU ===")
B = 2
batch = {
    img_key: torch.rand(B, 3, 512, 512, device=device),
    OBS_STATE: torch.rand(B, state_dim, device=device),
    OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (B, config.tokenizer_max_length), device=device),
    OBS_LANGUAGE_ATTENTION_MASK: torch.ones(B, config.tokenizer_max_length, dtype=torch.long, device=device),
    ACTION: torch.rand(B, config.chunk_size, action_dim, device=device),
}

print("=== forward pass on GPU ===")
with torch.no_grad():
    loss, loss_dict = policy.forward(batch)
print("loss:", float(loss))
print("loss_dict:", loss_dict)
assert torch.isfinite(loss), "loss is not finite!"
print("=== GPU SMOKE TEST PASSED ===")
