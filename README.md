# VLA 从零到部署

> 视觉-语言-动作大模型课程 — 从仿真到真机

[📖 在线阅读](https://howe12.github.io/vla-course) | [📋 课程大纲](COURSE_OUTLINE.md)

## 快速开始

```bash
git clone https://github.com/howe12/vla-course.git
cd vla-course/codes
uv sync
uv run python step2_sim/sim_vla_arm.py
```

## 实机部署文档

| 文档 | 内容 |
|---|---|
| [DM05_DEPLOYMENT_HANDOFF.md](docs/DM05_DEPLOYMENT_HANDOFF.md) | DM0.5 (OpenDM) 部署与实机对接交接文档 |
| [DM05_REALROBOT_DEBUG_NOTES.md](docs/DM05_REALROBOT_DEBUG_NOTES.md) | DM0.5 实机调试记录：抓取失败根因（起始位姿 OOD）、整机归位修复与实测数据 |
| [DM05_CLOUD_PATCH_BACKUP.md](docs/DM05_CLOUD_PATCH_BACKUP.md) | 云端 OpenDM 容器补丁备份清单 |
| [SMOLVLA_DEPLOYMENT_NOTES.md](docs/SMOLVLA_DEPLOYMENT_NOTES.md) | SmolVLA 实机部署笔记 |
| [USB_CAMERA_BANDWIDTH_CONCLUSION.md](docs/USB_CAMERA_BANDWIDTH_CONCLUSION.md) | 三路 USB 相机带宽结论 |

对应代码：`codes/step7_gemini/dm05/`（DM0.5）、`codes/step7_gemini/grpc/`（SmolVLA）。
