# 云端容器 Patch 备份清单（DM0.5 / OpenDM）

> 生成时间：2026-09-23 18:25
> 背景：云端容器 `<GPU 容器 SSH 端点>` SSH 端口拒绝连接（容器疑似停止/回收）。
> 本文档记录**系统盘上的所有修改**，以便容器重建后快速恢复。
>
> ⚠️ **数据盘（`/root/gpufree-data`，LVM PVC）通常持久化；系统盘（overlay）在容器重建时会丢失。**

---

## 一、系统盘上需要恢复的内容（有丢失风险）

### 1. OpenDM 源码 patch（3 处，关键）

仓库位置：`/root/opendm`（git clone 自 `https://github.com/dexmal/opendm`）

#### Patch 1 — `opendm/model/dm05/dm05_arch.py` `_real_init`

**问题**：SigLIP vision tower 在 `from_pretrained` 初始化时默认尝试 `flash_attention_2`，
但 flash-attn 未安装（L40 上编译 >40 分钟未完成），导致加载失败。

```python
    def _real_init(self, config: DM05Config):
        # Force vision tower to use sdpa when flash_attn is not installed
        if hasattr(config, 'vlm_config') and hasattr(config.vlm_config, 'vision_config'):
            if not is_flash_attention_2_available():
                config.vlm_config.vision_config._attn_implementation = "sdpa"
        self.model = DM05Model(config)
```

#### Patch 2 — `opendm/model/dm05/dm05_arch.py` `_resolve_vision_attn_implementation`

**问题**：请求 `flash_attention_2` 但不可用时直接 `raise ImportError`，不 fallback。

```python
        if requested == "flash_attention_2":
            if self.precision_policy != FP32_MIXED_PRECISION_POLICY and not bf16:
                raise ValueError("flash_attention_2 requires bf16=True")
            if not torch.cuda.is_available():
                raise RuntimeError("flash_attention_2 requires CUDA")
            if not is_flash_attention_2_available():
                logger.warning("flash_attention_2 requested but flash_attn not installed, falling back to sdpa")
                return "sdpa"
            return requested
```

#### Patch 3 — `opendm/trainer/trainer.py`（2 处）

**问题**：accelerate 1.14.0 在非 FSDP 模式下没有 `state.fsdp_plugin` 属性，直接访问抛 AttributeError。

```python
# 第 66 行附近（_configure_fsdp_precision）
fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
if fsdp_plugin is not None:

# 第 80 行附近（_configure_fsdp_auto_wrap）
fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
if fsdp_plugin is None:
```

### 2. 新增文件（系统盘）

#### `opendm/dataset/leo_gemini.py` — 数据集注册

```python
"""LEO-Gemini dataset registration for DM0.5 fine-tuning."""

from opendm.constants.robot import RobotStateDesc, RobotType
from opendm.dataset.register import register_dataset

LEO_STATE_DESC = (
    [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
)

register_dataset({
    "leo_gemini": {
        "jsonl_dir": "/root/gpufree-data/opendm/datasets/leo_gemini/",
        "image_dir": "/root/gpufree-data/opendm/datasets/leo_gemini/",
        "image_keys": ["images_1", "images_2", "images_3"],
        "image_prompts": ["Head", "Left wrist", "Right wrist"],
        "robot_type": RobotType.ALOHA,
        "state_desc": LEO_STATE_DESC,
        "fps": 30,
    },
})
```

#### `playground/dm05_leo_lora.py` — LEO 训练配置（LoRA + bf16 + sdpa，单卡 L40）

关键字段（其余继承 `_DM05*`）：

```python
@dataclass
class DM05DataConfig(_DM05DataConfig):
    dataset_name: str = field(default="leo_gemini")

@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    precision_policy: Literal["bf16_mixed", "fp32_mixed"] = field(default=BF16_MIXED_PRECISION_POLICY)
    llm_attn_implementation: Literal["auto","eager","sdpa","flex_attention"] = field(default="sdpa")
    vision_attn_implementation: Literal["auto","eager","sdpa","flash_attention_2"] = field(default="sdpa")
    vlm_gradient_checkpointing: bool = field(default=True)
    ae_gradient_checkpointing: bool = field(default=True)

@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    base_lr: float = field(default=1e-4)
    warmup_steps: int = field(default=500)

@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = field(default="/root/gpufree-data/opendm/user_checkpoints/leo_lora")
    fsdp1: bool | None = field(default=False)          # 单卡必须关闭
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    save_steps: int = field(default=500)
    num_train_steps: int = field(default=2000)
    save_only_model: bool = field(default=True)

@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=14)
    image_prompts: list[str] = field(default_factory=lambda: ["Head","Left wrist","Right wrist"])

@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=True)        # LoRA 必需（全量会 OOM）
    ...
```

#### `playground/dm05_leo_smoke.py` — 同上的 smoke test 版本（10 步）

#### `convert_lerobot_to_dm05.py` — lerobot v3.0 → DM05 JSONL+jpg 转换脚本

要点：
- 输入：`/root/gpufree-data/datasets/grasp_two_obj_20260907_141217`（30 episodes, 25851 帧）
- 输出：`/root/gpufree-data/opendm/datasets/leo_gemini/`
- **必须流式解码**（`av` 逐帧保存），一次性加载全部帧会吃 60GB+ 内存被 OOM kill
- 图像映射：`observation.images.front → images_1`，`left → images_2`，`right → images_3`
- JSONL 每行含 `state`（14 维）、`prompt`、`is_robot`、`images_1/2/3`

---

## 二、数据盘上的内容（通常持久化）

路径：`/root/gpufree-data/opendm/`

| 内容 | 说明 |
|---|---|
| `checkpoints/DM05/` | 基础模型 11.1GB（`model.safetensors` + `norm_stats.json`）|
| `conda_env/` | conda 环境（8.4GB），系统盘 `/opt/conda/envs/opendm` 软链接到此 |
| `user_checkpoints/leo_lora/checkpoint-500/` | **已训练的 LoRA 权重**（619MB，含 `norm_stats.json`）|
| `datasets/leo_gemini/` | 转换后的训练数据（30 episodes, 25851 帧, 3.1GB）|
| `caches/huggingface/`、`caches/pip/` | 缓存（系统盘 `~/.cache/*` 软链接到此）|
| `attrib_exp.py` 等 8 个分析脚本 | 归因实验脚本 |
| `nuc_*.jpg`、`open_*.jpg` | NUC 实拍对比图 |
| `*.log` | 训练/推理日志 |

### 恢复软链接（容器重建后需重新创建）

```bash
mkdir -p /root/gpufree-data/opendm
ln -sfn /root/gpufree-data/opendm/checkpoints /root/opendm/checkpoints
ln -sfn /root/gpufree-data/opendm/conda_env /opt/conda/envs/opendm
ln -sfn /root/gpufree-data/opendm/caches/huggingface /root/.cache/huggingface
ln -sfn /root/gpufree-data/opendm/caches/pip /root/.cache/pip
```

---

## 三、环境安装要点（容器重建后）

```bash
# 1. conda 渠道修复（清华镜像 pkgs/pro 已失效）
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro/

# 2. 创建环境
conda create -n opendm python=3.10 -y

# 3. 安装依赖（分步 + nohup，避免 SSH 超时）
pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==5.3.0 accelerate==1.14.0 peft==0.19.1 bitsandbytes==0.49.2 \
            tokenizers==0.22.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm==1.0.27 diffusers==0.38.0 datasets==5.0.0 decord av albumentations \
            -i https://pypi.tuna.tsinghua.edu.cn/simple
# ... 详见 /root/install_opendm.sh（若还在）
cd /root/opendm && pip install -e . --no-deps

# 4. 模型下载（hf-mirror + 禁用 xet）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
hf download Dexmal/DM05 --local-dir /root/gpufree-data/opendm/checkpoints/DM05
# 注意：新版 huggingface_hub 用 `hf` 命令，`huggingface-cli` 已弃用
```

### 已知安装坑

| 坑 | 解决 |
|---|---|
| `conda: command not found` | `source /opt/conda/etc/profile.d/conda.sh` |
| 清华 `pkgs/pro` 渠道 404 | 移除该渠道 |
| 大包 pip 下载超 10 分钟 | 用 `nohup` 后台跑，勿让 SSH 超时 |
| flash-attn 编译 >40 分钟 | **不要装**，用 sdpa 替代（见 Patch 1/2）|
| torch 已随 conda create 装好 | torch 2.11.0+cu128 会自动出现，勿重复安装 |
| 全量训练 batch=8 在 L40 OOM | 改用 **LoRA + bf16 + batch=1 + grad_accum=8** |

---

## 四、启动推理服务（恢复后）

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate opendm
cd /root/opendm
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup script/dm05_launcher.sh \
  --exp playground/dm05_leo_lora.py \
  --task inference \
  --model-config.model-name-or-path /root/gpufree-data/opendm/user_checkpoints/leo_lora/checkpoint-500 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts Head "Left wrist" "Right wrist" \
  --inference-config.port 7891 \
  > /root/gpufree-data/opendm/inference.log 2>&1 &
```

验证：`curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7891/` → 应返回 **404**（根路径无 handler，属正常）

---

## 五、关键实验结论（避免重复踩坑）

### 模型输出语义（已实测确认）

| 项目 | 结论 |
|---|---|
| 动作值域 | **度（degree）**，与 state 同量级，非归一化域 |
| 动作语义 | 训练 `action_mode=RELATIVE`（默认）；推理端 `ActionAbsolute` 做 `state + delta`，**gripper 维保持绝对值** |
| **关键** | **请求里的 `state` 参与最终动作计算 → state 错误会直接影响全部 50 步的绝对目标** |
| chunk | 50 步 @30fps = **1.667 秒**的示范轨迹 |
| HTTP API | **无时间戳字段**，服务端无状态，不校验观测时刻 |

### 分布特性（最重要的发现）

| 关节 | 训练数据覆盖 | 影响 |
|---|---|---|
| **gripper** | `<0`（闭合）仅 **34/25851 = 0.13%** | 夹爪在 −80 时模型输出 +6~+12（回归分布主体），偏差 86~92° |
| L_elbow | min −90.0°，P0.5 −90.0° | 模型可能要求到 −108°（越界 18~20°）|
| shoulder | L[−64.1, 95.1] R[−62.6, 104.5] | 悬停位（L≈86.8 R≈97.9）在范围内 |

**实测数据**：夹爪 −80 起步 → chunk[0] MAE **13.24°**；夹爪 +20 起步 → **0.67°**（20 倍改善）

### 已排除的假设（不要再查）

| 假设 | 实验 | 结果 |
|---|---|---|
| 图像域差异（桌面/背景） | 消融：NUC图/训练图/**灰图** | 差 **<0.11°** → 无关 |
| front 裁剪有害 | 原图/裁剪+拉伸/letterbox/裁剪2/3 | MAE 0.64~0.86 → 无关 |
| 开环长度 N | N=5 vs N=50（各 150 步）| N=50 更优（delta0 2.36 vs 4.21）→ 无关 |
| 限幅过严 | chunk 内部速度分析 | 仅 0.24°/步，6.9% 超速 → 非主因 |
| 起始姿态 OOD | NN 距离 | = 训练第 0 集首帧（14.4°）→ 正常 |
| 时间戳不匹配 | 协议检查 | API 无时间戳字段 → 不存在该问题 |

### 训练性能实测（L40 46GB）

| 项目 | 数值 |
|---|---|
| LoRA 可训练参数 | 324M / 5.83B（5.27%）|
| 每步耗时 | ~8.2 s（batch=1, grad_accum=8）|
| 1000 步 loss | 0.3641 → 0.0156（step 500）|
| 2000 步预计 | ~4.6 小时 |
| LoRA checkpoint 大小 | 619 MB |

---

## 六、内存坑（转换数据集时）

`convert_lerobot_to_dm05.py` 若一次性把所有视频帧解码到内存，会占用 **60GB+ 内存**并被 OOM kill。
必须**流式**处理：逐帧 `container.decode()` → `img.save()` → 释放。

---

*本文件由 DM0.5 部署会话生成，用于容器重建后的快速恢复。*
