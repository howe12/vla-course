# DM0.5 LoRA 重训报告（v3，2026-09-28）

> 日期：2026-09-28
> 目的：修「欠训练」（旧 checkpoint-500 仅 0.155 epoch）导致的右臂外推/偏下
> 平台：云端 L40（46GB 单卡）+ 同数据（leo_gemini_new，30 集 25843 帧）重训

---

## 一、关键发现

1. **旧训练只跑了 0.155 epoch**：500 步 × 有效批 8 = 4000 样本 / 25843 ≈ 0.155 epoch，
   LR 已按 500 步退火到底 —— 是**欠训练**，不是收敛。
2. **单卡必须 `--nproc_per_node 1`**：launcher 默认 8，会导致 8 个进程各加载
   整份 5.83B 模型 → CUDA OOM（46GB 装不下）。旧成功训练日志确认为 nproc=1。
3. **`save_only_model=True` ⇒ checkpoint 无优化器/调度器状态，不可中断续训**；
   若要断点续练，需扩展 exp 支持「从 adapter 初始化 + 新 LR 调度」。
4. tyro bool 参数必须 `=False` 形式（`--model-config.xxx=False`），裸 `False` 报
   Unrecognized options（踩过坑，浪费两次启动）。

## 二、参数与理由（v3 配置）

| 参数 | 值 | 理由 |
|---|---|---|
| 任务/实验 | `--task train --exp playground/dm05_leo_lora_new.py` | 同旧训练 |
| 基座 | `--model-config.model-name-or-path ./checkpoints/DM05` | 5.83B Gemma3 系 |
| nproc | `--nproc_per_node 1` | **单卡必须**（见发现 2） |
| 总步数 | `--trainer-config.num-train-steps 3200` | 1 epoch = 3230 步（25843/8） |
| 有效批 | `per-device-train-batch-size 4` × `gradient-accumulation-steps 2` | **=8 与旧 run 一致，LR 语义不变**；bs 1→4 摊薄固定开销 |
| 精度 | bf16 + sdpa（默认） | flash/flex_attention 会 InductorError，别碰 |
| LoRA | r=32, alpha=16, all-linear + time 模块（默认） | 旧配置，324M 可训练（5.27%） |
| 学习率 | base_lr 1e-4, warmup 50, cosine→min_lr(0.1×) | LoRA 常见量级，与旧一致 |
| 存档 | save_steps 500, save_total_limit 8, save_only_model=True | 离线回放评选需要中间档 |
| norm_stats | `--data-config.norm-stats-root <旧checkpoint目录>` | 数据没变，跳过全量重算（省大量时间） |

## 三、实测速度验收（关键）

```
旧 bs1×accum8: 8.15 s/opt-step（0.978 样本/s）
新 bs4×accum2: 4.50 s/opt-step（GPU 利用率 100%，显存 19GB/46GB）  → +45%
3200 步 ≈ 4.0 小时（~19:30 完成）
```

- 说明：1→4 batch 摊薄了 Python/内核/数据加载固定开销；再往上 bs 收益递减（计算量固定）。
- 数据配置默认 `dataloader_num_workers=4` + prefetch，已并行取帧。

## 四、Loss 轨迹（每百步采样）

| step | loss | fm_loss |
|---|---|---|
| 10 | 0.1559 | 0.1682 |
| 100 | 0.0420 | 0.0529 |
| 500 | 0.0226 | 0.0176 |
| 800 | 0.0136 | 0.0386 |
| 1000 | 0.0144 | 0.0145 |
| 1600 | 0.0120 | 0.0035 |

对照旧 run：500 步终点 loss 0.023（LR 已退火到底）；新 run 0.46 epoch 时 loss 0.012，
且 LR 尚在高位 → 训练方向正确、明显优于旧模型。

## 五、离线回放验收（不动电机，几分钟）

把数据集自身图像喂 `/v1/infer`，预测 vs 真实 `state[k+1..k+50]` 的 MAE：

| 帧（臂状态） | 旧 ckpt-500 | 新 ckpt-1500 | 提升 |
|---|---|---|---|
| 帧 0（初始位姿） | 1.56° | 1.51° | ≈ |
| 帧 100（左臂下探中） | 9.20° | **1.13°** | 8.1× |
| 帧 280（左臂深探） | 1.55° | 1.11° | 1.4× |
| 帧 600（右臂深探） | 2.62° | **1.62°** | 1.6×（R_shdr −24.2 vs 真值 −24.8） |
| 帧 700（右臂深探中） | 6.07° | **2.26°** | 2.7× |

⇒ **诊断工具值得沉淀**：任何 checkpoint / 映射改动的离线回放 MAE（脚本在
`/tmp/loss_track.py` 与本文方法），可作为实机前的第一道闸。

## 六、实机验证（30 轮，v3 ckpt-1500 + 正确映射 + SAFE 限位）

- 右臂不再越界：初始 4/60 轮 `R_shdr < -39°`（最低 −44.1）→ **0/30**（最低 −30.4）
- 夹爪全行程：左 10.0~59.5 / 右 12.4~59.2 ✓；臂交替结构保持（先左后右）✓
- 随后发现「右臂偏下」第二元凶 = **video2 静态帧**（见调试笔记 §0.2-3），
  加新鲜帧确认后 0/15 轮静态帧、右臂走出训练分布轨迹
- 注：本次用 `--limits safe`（SAFE_LIMITS）；`envelope` 实验见调试笔记 §0.2-5（弃用）

## 七、下一步

1. 修复新鲜帧后跑完整 60 轮验证两臂时序与抓取闭环
2. 若需续训：方案 A（adapter 初始化 + 新调度，不能 HF resume）
3. 长期：重采时确认物理相机标签 + 随机化物体摆放以提升泛化