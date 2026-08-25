# Sagittarius SGR532 机械臂 MuJoCo 接入（02-3/02-4）

Sagittarius（射手座）机械臂在 MuJoCo 中的显示与控制。源模型来自 https://github.com/howe12/sagittarius_humble_ws（`sagittarius_descriptions`），已转成 MuJoCo 可加载的 MJCF。

## 文件清单

| 文件 | 说明 |
|------|------|
| `sgr532.mjcf` | Sagittarius 的 MuJoCo 模型（6 关节 J1-J6 + 夹爪 GL/GR） |
| `sgr532_verify.py` | 验证脚本：加载 + 离屏渲染 + 关节控制 + 物理步进 |
| `meshes/*.obj` | 连杆 STL 转 OBJ 网格（trimesh 转换） |

## 运行（L40, CPU 无需 GPU）

```bash
cd /root/gpufree-data/vla-course/sim_course/02_sim/sagittarius
/root/gpufree-data/vla-course/codes/.venv/bin/python sgr532_verify.py
```

输出：`sgr532_home.png` / `sgr532_expand.png`（480×480, MuJoCo 离屏 OSMesa 渲染，白色仿真房间 + 墙装相机视角）。

## 模型结构（SGR532）

- 6 自由度关节臂 `J1..J6`（revolute）+ 夹爪 `GL/GR`（slide）
- `nq=8`，层级 `base_link → link1..link6 → grasp → gripper_left/right`
- MJCF 由 `sagittarius_descriptions/urdf/sgr532.urdf.xacro` 的关节几何解析重构

## 已知说明

- 源仓库的 `link_grasping_frame` mesh 为空文件（84B），末端连杆用几何胶囊手绘替代（`fg_main/fg_arm_l/fg_arm_r`）
- 相机 `arm_cam` 固定在 +y 内墙壁面中心，`mode="targetbody" target="link6"` 朝向机械臂
- 白色仿真环境 = 白色地板 + 四面白墙（~1.6m 小房间）