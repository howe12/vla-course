# Sagittarius SGR532 机械臂 MuJoCo 接入（02-3/02-4）

Sagittarius（射手座）机械臂在 MuJoCo 中的显示与控制。源模型来自 https://github.com/howe12/sagittarius_humble_ws（`sagittarius_descriptions`），已转成 MuJoCo 可加载的 MJCF。

## 文件清单

| 文件 | 说明 |
|------|------|
| `sgr532.mjcf` | Sagittarius 的 MuJoCo 模型（6 关节 J1-J6 + 夹爪 GL/GR） |
| `sgr532_motion.mjcf` | 控制版模型：加了 8 个 position 伺服 actuator，隐式积分防 stiff 爆 NaN |
| `sgr532_verify.py` | 验证脚本：加载 + 离屏渲染 + 关节控制 + 物理步进 |
| `sgr532_control.py` | 运动控制 demo：姿态序列 smoothstep 插值 + 位置伺服，渲染运动 GIF |
| `sgr532_teleop.py` | 键盘遥操作 + 夹爪开合 + 轨迹录制（供数据采集），交互/--terminal/--demo |
| `meshes/*.obj` | 连杆 STL 转 OBJ 网格（trimesh 转换） |

## 运行（L40, CPU 无需 GPU）

```bash
cd /root/gpufree-data/vla-course/sim_course/02_sim/sagittarius
/root/gpufree-data/vla-course/codes/.venv/bin/python sgr532_verify.py
```

输出：`sgr532_home.png` / `sgr532_expand.png`（480×480, MuJoCo 离屏 OSMesa 渲染，白色仿真房间 + 墙装相机视角）。

## 运动控制与键盘遥操作（02-4 / 数据采集）

```bash
# 运动控制 demo: 姿态序列插值 + 位置伺服, 渲染 GIF
/root/gpufree-data/vla-course/codes/.venv/bin/python sgr532_control.py     # -> sgr532_motion.gif

# 键盘遥操作(有窗口的完整 mujoco 机器): 6 关节 + 夹爪 + 空格录制 + k 保存
python sgr532_teleop.py
# 无窗口终端遥操作(任意机器/SSH): 键位一样, Ctrl-C 退出自动保存轨迹 CSV
/root/gpufree-data/vla-course/codes/.venv/bin/python sgr532_teleop.py --terminal
# 无头自检记录链路
/root/gpufree-data/vla-course/codes/.venv/bin/python sgr532_teleop.py --demo
```

- 键位：`q/a` J1± `w/s` J2± `e/d` J3± `r/f` J4± `t/g` J5± `y/h` J6±，`z` 夹爪张开 `c` 闭合，空格录制开/停，`k` 保存轨迹到 `sgr532_traj_<ts>.csv`（表头 `t,J1..J6,GL,GR`）
- 控制版模型 `sgr532_motion.mjcf` 用 `integrator="implicit"` + kp30~40/kv5~6，位置伺服平滑跟位（稳态误差约 0.1rad，天然教学点）

## 模型结构（SGR532）

- 6 自由度关节臂 `J1..J6`（revolute）+ 夹爪 `GL/GR`（slide）
- `nq=8`，层级 `base_link → link1..link6 → grasp → gripper_left/right`
- MJCF 由 `sagittarius_descriptions/urdf/sgr532.urdf.xacro` 的关节几何解析重构

## 已知说明

- 源仓库的 `link_grasping_frame` mesh 为空文件（84B），末端连杆用几何胶囊手绘替代（`fg_main/fg_arm_l/fg_arm_r`）
- 相机 `arm_cam` 固定在 +y 内墙壁面中心，`mode="targetbody" target="link6"` 朝向机械臂
- 白色仿真环境 = 白色地板 + 四面白墙（~1.6m 小房间）