# 通用机械臂 LeRobot SO-101 MuJoCo 接入（02-3 通用型号）

SO-101 是 LeRobot 官方生态的通用教学机械臂（5 自由度假 PLUS 夹爪），我们把它和自研 Sagittarius SGR532 用**同一套流程**接入 MuJoCo，体现"换一台臂也能照做"。源模型来自本机 `Arm/Lerobot_ros2`（ROS2 Humble，`so101_follower_description` 包，描述文件 `so101_follower.urdf.xacro`），已转成 MuJoCo 可加载的 MJCF。

## 文件清单

- `so101.mjcf` — SO-101 的 MuJoCo 模型（6 关节 shoulder_pan/lift、elbow_flex、wrist_flex/roll、gripper）
- `so101_verify.py` — 验证脚本：加载 + 离屏渲染 + 关节控制 + 物理步进
- `meshes/*.obj` — 连杆 STL 转 OBJ 网格，共 13 个（trimesh 转换；`wrist_roll_follower` 原 mesh 为毫米单位，asset 里标 `scale=0.001` 还原）

## 运行（L40, CPU 无需 GPU）

```bash
cd /root/gpufree-data/vla-course/sim_course/02_sim/so101
/root/gpufree-data/vla-course/codes/.venv/bin/python so101_verify.py
```

输出：`so101_home.png` / `so101_expand.png`（480×480, MuJoCo 离屏 OSMesa 渲染，白色仿真房间 + 墙装相机视角）。

## 模型结构（SO-101）

- 关节链 `base_link → shoulder_link → upper_arm_link → lower_arm_link → wrist_link → gripper_link → jaw_link`
- 6 个 revolute 关节（含夹爪 `gripper`），`nq=6`，全用 Feetech STS3215 舵机
- MJCF 由 `so101_follower.urdf.xacro` 的几何 + 材质（`3d_printed` 黄 / `sts3215` 深灰）解析重构
- 末端 `wrist_roll_follower` 原 STL 是毫米单位（URDF 里带 `scale=0.001`），其余均为米制

## 已知说明

- 相机 `arm_cam` 固定在 +y 内墙壁面中心，`mode="targetbody" target="jaw_link"` 朝向末端
- 白色仿真环境 = 白色地板 + 四面白墙（~1.6m 小房间），与 SGR532 完全一致
- 夹爪为单侧活颚（`moving_jaw`），由 `gripper` revolute 关节驱动