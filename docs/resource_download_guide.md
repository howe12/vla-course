# 课程资源下载指南

> 适用于《VLA 策略复刻》课程。所有资源从 Datawhale 公开仓库获取。

## 1. OMY 仿真模型（MuJoCo 场景文件 + 机械臂 + 物体）

### 来源

Datawhale every-embodied 仓库 `06-策略抓取或抓取VLA/` → `asset/` 目录。

### 方法一：Git 稀疏检出（推荐）

```bash
# 只拉 asset 目录，不下载整个 16GB 仓库
cd /tmp
mkdir every_embodied_assets && cd every_embodied_assets
git clone --depth 1 --filter=blob:none --sparse \
    https://gitcode.com/gh_mirrors/ev/every-embodied.git
cd every-embodied
git sparse-checkout set "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/asset"
git checkout

# 复制到课程目录
cp -r "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/asset/"* /workspace/vla-course/models/omy/
```

> **备用源**：如果 gitcode 不可用，尝试 `https://github.com/datawhalechina/every-embodied.git`

### 方法二：直接下载（快速，约 87MB）

从 every-embodied 仓库的 asset 目录逐个下载文件，然后解压 plate_11.zip：

```bash
# 关键文件清单（共6个 include 路径）：
#   tabletop/object/floor_isaac_style.xml
#   tabletop/object/object_table.xml
#   robotis_omy/omy.xml (+ assets/)
#   objaverse/mug_5/model_new.xml  (+ visual/, collision/)
#   objaverse/plate_11/model_new.xml  ← 需解压 plate_11.zip
#   objaverse/mug_6/model_new.xml  (+ visual/, collision/)
```

### 整合到课程仓库后的目录结构

```
vla-course/models/omy/
├── example_scene_y2.xml          ← 主场景文件
├── robotis_omy/
│   ├── omy.xml
│   └── assets/                    ← STL 网格文件
├── tabletop/
│   ├── mesh/                      ← 纹理贴图
│   └── object/
│       ├── floor_isaac_style.xml
│       └── object_table.xml
└── objaverse/
    ├── mug_5/
    │   ├── model_new.xml          ← 红杯
    │   ├── visual/
    │   └── collision/
    ├── mug_6/
    │   ├── model_new.xml          ← 蓝杯
    │   ├── visual/
    │   └── collision/
    └── plate_11/
        ├── model_new.xml          ← 盘子
        ├── visual/
        └── collision/
```

---

## 2. 训练数据集 (Datawhale 公开数据集)

### 来源

HuggingFace: `Datawhale/datawhale_eai_pnp_language`（382 MB，20 episodes，LeRobot v2.1 格式，Apache 2.0 许可）

### 方法一：huggingface_hub（推荐，支持断点续传）

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像加速

from huggingface_hub import snapshot_download
snapshot_download(
    'Datawhale/datawhale_eai_pnp_language',
    repo_type='dataset',
    local_dir='./demo_data_language',
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
)
```

### 方法二：Git LFS（如果容器已安装 git-lfs）

```bash
GIT_SSL_NO_VERIFY=1 git clone https://hf-mirror.com/datasets/Datawhale/datawhale_eai_pnp_language
cd datawhale_eai_pnp_language
git lfs pull
```

### 方法三：浏览器手动下载

访问 https://hf-mirror.com/datasets/Datawhale/datawhale_eai_pnp_language → 点击 "Files and versions" → 下载所有文件。

---

## 3. 格式转换（v2.1 → v3.0）

> ⚠️ Datawhale 公开数据集为 LeRobot v2.1 格式。如果你使用 LeRobot 0.6.0+，需要转换为 v3.0。

```bash
# 方案A：使用 LeRobot 内置转换脚本
source /workspace/venv/bin/activate
python3 -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id Datawhale/datawhale_eai_pnp_language \
    --root /tmp/demo_data_language \
    --push-to-hub false

# 方案B：手动轻量转换（仅改 codebase_version，适合已下载的本地数据）
cp -r /tmp/demo_data_language /tmp/demo_data_language_v30
python3 -c "
import json
with open('/tmp/demo_data_language_v30/meta/info.json') as f:
    info = json.load(f)
info['codebase_version'] = 'v3.0'
with open('/tmp/demo_data_language_v30/meta/info.json', 'w') as f:
    json.dump(info, f, indent=2)
"
```

> **注意**：如果 LeRobot 转换脚本报 `HFValidationError`（本地路径不是 HF repo），使用方案B。

---

## 4. 环境对齐验证

下载完成后运行审计脚本确认数据格式正确：

```bash
cd /workspace/vla-course
python3 dw_reference/07_audit_data.py --dataset ./demo_data_language
```

预期输出：20 episodes，2621 frames，action 为 7-D（6 关节 + 1 夹爪）。

---

## 注意事项

- **网络问题**：国内环境建议使用 `hf-mirror.com`，或配置 HTTP 代理
- **磁盘空间**：OMY 模型约 90MB，数据集约 382MB，合计不到 500MB
- **plate_11.zip** 是唯一的压缩文件，需手动解压：`unzip plate_11.zip -d .`
- 如果 hugingface_hub 报 `ConnectError: Network is unreachable`，尝试：
  ```python
  os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
  ```
