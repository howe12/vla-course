# vla-course / sim-course 分支

VLA 仿真课程（飞书 wiki 对应章节）。主线 LeRobot + MuJoCo + 我司 Sagittarius 机械臂。

## 分支约定

| 分支 | 用途 |
|------|------|
| `main` | 旧 VLA 教程（OpenVLA + LIBERO，step1-7） |
| `sim-course` | 本课程：仿真课程代码（MuJoCo + LeRobot + Sagittarius） |
| `real-course` | 后续：真机部分（规划中，未建） |

## 环境

- 云端：L40（120.209.70.195, 8×48GB），容器 30334
- Python venv：`codes/.venv`（mujoco 3.10 / torch 2.10+cu128 / lerobot 0.4.4）
- 机械臂：Sagittarius https://github.com/howe12/sagittarius_humble_ws

## 目录结构

```
sim_course/
├── 01_panorama/    # 章一 VLA 全景（纯概念）
├── 02_sim/         # 章二 仿真环境与 L40 + MuJoCo + 机械臂
│   └── sagittarius/# Sagittarius 机械臂接入
├── 03_data/        # 章三 数据采集与数据集
├── 04_act/         # 章四 ACT
├── 05_smolvla/     # 章五 SmolVLA
├── 06_pi0/         # 章六 Pi0
├── 07_dm05/        # 章七 原力灵机 DM0.5
└── 08_robodojo/    # 章八 RoboDojo 统一评测
```

## 使用

```bash
# L40 服务器
cd /root/gpufree-data/vla-course
git checkout sim-course
source codes/.venv/bin/activate  # 或 codes/.venv/bin/python xxx.py

# 环境自检（CPU，无需 GPU）
python sim_course/env_check.py
```