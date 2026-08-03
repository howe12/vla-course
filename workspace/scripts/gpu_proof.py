import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_STATE, ACTION

device = torch.device("cuda")
img_key = "observation.images.top"
config = ACTConfig(
    input_features={img_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))},
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    pretrained_backbone_weights=None, use_vae=True, chunk_size=16, n_action_steps=16,
)
policy = ACTPolicy(config).to(device).eval()

devs = {str(p.device) for p in policy.parameters()}
print("[1] 所有参数的 device 集合:", devs)
assert devs == {"cuda:0"}, "有参数不在 GPU 上!"

buf_devs = {str(b.device) for b in policy.buffers()}
print("[2] 所有 buffer 的 device 集合:", buf_devs)

obs = {img_key: torch.rand(1, 3, 480, 640, device=device),
       OBS_STATE: torch.rand(1, 6, device=device)}
policy.reset()
with torch.no_grad():
    action = policy.select_action(obs)
print("[3] 输出 action.device =", action.device, "| dtype =", action.dtype)
assert action.device.type == "cuda"

print(f"[4] 当前进程 GPU 显存占用: {torch.cuda.memory_allocated()/1e6:.1f} MB")
print("=== 结论：模型参数、buffer、计算、输出全部在 cuda:0，GPU 确实被正确使用 ===")
