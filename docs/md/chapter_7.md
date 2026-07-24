# 第 7 章：Gemini 真机部署

> 将云端训练的 VLA 模型部署到双子座机器人上验证。

---

> **🔄 从 §5-§6 的云端训练出发** SmolVLA 和 Pi0 在 L40 上训好了，checkpoint 躺在 `/root/gpufree-data/vla-course/outputs/` 里。现在把这些"数字大脑"装进**真实的 Gemini 双臂机器人**------让仿真里验证过的 VLA 模型，驱动 4 条实体机械臂完成操作任务。

Gemini 真机部署
---------------

将云端训练的 VLA 模型部署到双子座机器人上验证。

> **🎯 学习目标**
>
> -   理解 Gemini 双臂机器人的硬件架构（4 臂 + NUC + 4 相机）
> -   掌握 VLA 模型从 L40 到边缘设备的转换流程（PT→ONNX→TensorRT→INT8）
> -   配置 LeRobot 真机接口，搭建 4 路相机实时推理管线
> -   理解 sim2real gap 的三大来源（视觉域差、物理参数差、延迟差）
> -   实现安全防护：力限制、速度钳制、急停

> **📍 本章定位**
>
> 本章是**VLA 课程的落地章**。前 6 章在仿真和云端完成了"理解→推理→训练"，本章把一切搬进物理世界。你在这里会第一次看到------仿真里那个正弦摆动的机械臂，变成了在你面前执行"把杯子放到桌上"的真实金属手臂。这是每个机器人工程师最兴奋的时刻。

<div>

### 7.1 Gemini 硬件架构

Gemini（双子座）是 NXROBO 的双臂机器人平台。先理解它在物理世界里的"身体"：

  组件           规格                                  VLA 中的角色
  -------------- ------------------------------------- ----------------------------------
  **机械臂**     4 臂（1 主 3 从），6-DOF + 夹爪       执行 VLA 输出的动作
  **计算单元**   NUC i7-14650HX, 20 核, RTX 4060 8GB   运行 VLA 推理
  **相机**       4 路 RGB 相机（前/左/右/腕部）        VLA 的"眼睛"------提供视觉观测
  **力传感器**   各关节力矩传感器                      安全监控 + 力控模式
  **主从控制**   1 条主臂可遥操作，3 条从臂跟随        数据采集（模仿学习前置课程）

#### 与 L40 仿真环境的差异

  维度       L40 仿真（MuJoCo/LIBERO）   Gemini 真机
  ---------- --------------------------- ---------------------------------------
  **视觉**   无噪完美 RGB，固定光照      真实光照、运动模糊、相机畸变
  **物理**   理想刚体，无摩擦误差        关节间隙、摩擦力、重力补偿误差
  **延迟**   仿真步长 0.002s             相机读取 30ms + 推理 40ms + 通信 10ms
  **安全**   碰撞 = 重来                 碰撞 = 损坏硬件或伤人

> **⚠️ ⚠️ sim2real gap**
>
> 仿真里训到 90% 成功率的模型，到真机上可能只有 30%。这不是代码 bug------是物理世界比数学方程复杂得多。本章的安全机制和域适应策略，就是为缩小这个 gap 服务的。

</div>

<div>

### 7.2 模型转换：从 L40 到 NUC

L40 上训好的 PyTorch 模型（400M-3B 参数），需要经过三次转换才能在 Gemini 的 RTX 4060（8GB）上高效运行：

    PyTorch (.pt) → ONNX → TensorRT → INT8 量化
     14GB 显存      通用格式    优化引擎    5-8GB 显存
     (L40 训练)                          (NUC 推理)

#### 转换流程

    # Step 1: PyTorch → ONNX（在 L40 上执行）
    import torch

    model = SmolVLA()
    model.load_state_dict(torch.load("outputs/smolvla/best.pt"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    torch.onnx.export(
        model, dummy_input, "smolvla.onnx",
        input_names=["image"], output_names=["action"],
        dynamic_axes={"image": {0: "batch"}},
        opset_version=17,
    )
    print("✅ ONNX 导出完成: smolvla.onnx")

    # Step 2: ONNX → TensorRT + INT8（在 NUC 上执行）
    trtexec --onnx=smolvla.onnx \
            --saveEngine=smolvla_int8.engine \
            --int8 \
            --fp16 \
            --workspace=4096

    # 验证
    # 预期: 模型大小 ~5GB, 推理延迟 ~20-40ms

#### 为什么需要 INT8？

  精度       模型大小   推理延迟   精度损失
  ---------- ---------- ---------- ------------
  **FP32**   \~14GB     \~80ms     0%（基准）
  **FP16**   \~7GB      \~50ms     \<0.1%
  **INT8**   \~3.5GB    \~25ms     \<1%

Gemini 的 RTX 4060 只有 8GB 显存------操作系统占 1GB，剩下 7GB 给模型。FP16 刚好够（但没余量做 batch），INT8 有 3.5GB 余量可以同时跑 2 个模型（左右臂各一个）。

</div>

<div>

### 7.3 推理管线

真机推理管线和 §4.4 的 LIBERO 仿真管线原理相同，但多了三个物理世界的挑战：多相机同步、实时通信、安全校验。

#### 完整管线图

    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ 相机 1   │   │ 相机 2   │   │ 相机 3   │   │ 相机 4   │
    │ (前方)   │   │ (左侧)   │   │ (右侧)   │   │ (腕部)   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
         │               │               │               │
         └───────────────┴───────┬───────┴───────────────┘
                                 │
                        ┌────────▼────────┐
                        │  图像预处理      │
                        │  · resize 224    │
                        │  · 拼接/选主     │
                        │  · 归一化        │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  VLA 推理        │
                        │  · TensorRT INT8 │
                        │  · 20-40ms       │
                        │  · 输出: 7 维动作│
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  安全检查        │
                        │  · 力限制        │
                        │  · 速度钳制      │
                        │  · 急停检测      │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  IK + 关节控制   │
                        │  · 动作→关节角   │
                        │  · 100Hz 发送    │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  🦾 Gemini 执行  │
                        └─────────────────┘

#### 延迟预算

  阶段           耗时                     累计
  -------------- ------------------------ -----------
  相机采集       \~30ms（4 路同步）       30ms
  图像预处理     \~3ms（resize + 拼接）   33ms
  **VLA 推理**   **\~25ms（INT8）**       **58ms**
  安全检查       \~1ms                    59ms
  IK + 通信      \~5ms                    64ms
  **总延迟**     **\~64ms**               **≈15Hz**

> **💡 15Hz 够用吗？**
>
> 对于桌面操作任务（抓取、放置、推拉），15Hz 足够。工业装配需要 50Hz+，但那是传统控制器的领域。VLA 的定位是**理解+规划**------15Hz 的推理频率配上 100Hz 的底层 IK 跟踪，可以覆盖 90% 的日常操作场景。

</div>

<div>

### 7.4 LeRobot 真机适配

LeRobot 是 HuggingFace 的机器人框架，提供了标准的真机接口。适配 Gemini 需要配置三个关键参数：

#### 运动学配置

    # configs/gemini.yaml
    robot:
      type: gemini
      arms: 4                    # 1 主 + 3 从
      dof_per_arm: 6             # 6 个关节
      gripper: binary            # 二值夹爪（开/关）
      
      # 关节限位（弧度）
      joint_limits:
        j1: [-3.14, 3.14]       # 基座旋转
        j2: [-2.0, 1.5]          # 肩部
        j3: [-2.5, 1.5]          # 肘部
        j4: [-3.14, 3.14]        # 腕旋转
        j5: [-2.0, 2.0]          # 腕俯仰
        j6: [-3.14, 3.14]        # 末端
      
      # 控制参数
      control_freq: 100           # 底层控制频率 (Hz)
      vla_freq: 15                # VLA 推理频率 (Hz)
      
      # 安全
      max_torque: 10.0            # 最大力矩 (Nm)
      max_velocity: 3.0           # 最大角速度 (rad/s)
      emergency_stop_torque: 15.0 # 急停阈值

#### LeRobot 接口实现

    from lerobot.robots import Robot

    class GeminiRobot(Robot):
        """Gemini 双臂机器人 LeRobot 适配器"""
        
        def __init__(self, config_path="configs/gemini.yaml"):
            self.config = self._load_config(config_path)
            self.arms = {}  # {"main": Arm, "slave1": Arm, ...}
            self.cameras = {}  # {"front": Camera, "left": Camera, ...}
            self._connect_hardware()
        
        def get_observation(self):
            """获取 VLA 推理所需的观测
            
            Returns:
                dict: {
                    "images": {"front": (224,224,3), "wrist": (224,224,3)},
                    "joint_state": {"main": [6], "slave1": [6], ...},
                    "gripper": {"main": 0.0, ...}  # 0=关, 1=开
                }
            """
            obs = {}
            obs["images"] = {name: cam.read() for name, cam in self.cameras.items()}
            obs["joint_state"] = {name: arm.get_joints() for name, arm in self.arms.items()}
            obs["gripper"] = {name: arm.get_gripper() for name, arm in self.arms.items()}
            return obs
        
        def execute_action(self, action, arm_name="main"):
            """执行 VLA 输出的动作
            
            Args:
                action: (7,) numpy — 6D 位姿增量 + gripper
                arm_name: 执行动作的机械臂
            """
            arm = self.arms[arm_name]
            
            # 1. 安全检查
            action = self._safety_check(action)
            
            # 2. 增量 → 绝对位姿
            current_pose = arm.get_end_effector_pose()
            target_pose = current_pose + action[:6] * 0.01  # 缩小步长
            
            # 3. IK → 关节角
            joint_angles = arm.inverse_kinematics(target_pose)
            
            # 4. 发送关节命令
            arm.set_joints(joint_angles)
            
            # 5. 夹爪控制
            if action[6] > 0:
                arm.open_gripper()
            else:
                arm.close_gripper()

</div>

<div>

### 7.5 安全机制

仿真可以随便 crash，真机不行。Gemini 的安全设计分三层：

  层级            机制           触发条件                响应
  --------------- -------------- ----------------------- ----------------------
  **L1 软限制**   动作钳制       VLA 输出超出安全范围    clip 到安全区间
  **L2 力保护**   力矩监控       任一关节力矩 \> 15 Nm   立即停止该臂
  **L3 急停**     物理急停按钮   人工按下                断电，所有臂立刻制动

#### L1：动作钳制

    def _safety_check(self, action):
        """VLA 输出的第一道防线"""
        # 位置增量限制（每步最大 5cm）
        action[:3] = np.clip(action[:3], -0.05, 0.05)
        # 旋转增量限制（每步最大 15°）
        action[3:6] = np.clip(action[3:6], -0.26, 0.26)
        # 夹爪不强制
        action[6] = np.clip(action[6], -1.0, 1.0)
        return action

#### sim2real 的三种 gap

  Gap 类型         来源                                     缓解策略
  ---------------- ---------------------------------------- --------------------------------------------
  **视觉域差**     仿真无噪完美图像 vs 真实光照+模糊+畸变   训练时加颜色抖动、高斯噪声、随机裁剪
  **物理参数差**   质量/摩擦/阻尼不准确                     域随机化：训练时随机扰动质量±20%、摩擦系数
  **延迟差**       仿真 500Hz vs 真机 15Hz                  训练时人为加入动作延迟（丢帧）

</div>

<div>

### 7.6 动手实验：SmolVLA 真机部署

把 L40 上训好的 SmolVLA checkpoint 部署到 Gemini 上运行端到端 demo：

> **🧪 🧪 Gemini 真机部署全流程**
>
> 1.  **L40 导出模型**：`uv run python codes/step7_gemini/deploy_model.py --ckpt outputs/smolvla/best.pt`
> 2.  **传输到 NUC**：`scp smolvla_int8.engine gemini@nuc-ip:/home/gemini/models/`
> 3.  **NUC 上加载模型**：`uv run python codes/step7_gemini/run_gemini.py --model smolvla_int8.engine`
> 4.  **观察**：终端打印每步动作 + 力矩监控，机械臂根据指令执行操作

展开查看 deploy\_model.py（模型转换脚本）


    #!/usr/bin/env python3
    """L40 → NUC 模型转换: PyTorch → ONNX → TensorRT INT8"""
    import torch, argparse, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step5_smolvla"))
    from train import SmolVLA

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--ckpt", default="outputs/smolvla/best.pt")
        parser.add_argument("--output", default="smolvla.onnx")
        args = parser.parse_args()

        print(f"加载 checkpoint: {args.ckpt}")
        model = SmolVLA().cuda()
        state = torch.load(args.ckpt, map_location="cuda")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()

        print("导出 ONNX...")
        dummy = torch.randn(1, 3, 224, 224).cuda()
        torch.onnx.export(
            model, dummy, args.output,
            input_names=["image"], output_names=["action"],
            opset_version=17,
        )
        print(f"✅ ONNX 导出: {args.output}")
        print("下一步: 复制到 NUC → trtexec --onnx=smolvla.onnx --int8 --saveEngine=smolvla_int8.engine")

    if __name__ == "__main__":
        main()


<div>

### 📝 本章小结

> **带走这几条**
>
> 1.  **Gemini = 4 臂 + NUC + 4 相机**，RTX 4060 8GB 限制模型必须 INT8 量化
> 2.  **模型转换三阶段**：PyTorch(.pt)→ONNX→TensorRT→INT8（14GB→3.5GB）
> 3.  **推理管线总延迟 \~64ms ≈ 15Hz**，VLA 做理解+规划，底层 100Hz IK 做跟踪
> 4.  **三层安全**：动作钳制(L1)→力矩监控(L2)→急停(L3)，仿真 crash vs 真机损坏
> 5.  **sim2real 三种 gap**：视觉域差（加噪声训练）、物理参数差（域随机化）、延迟差（丢帧训练）

#### 🤔 思考题

1.  Gemini 的 RTX 4060 只有 8GB 显存。如果未来需要同时跑 SmolVLA（左臂）+ SmolVLA（右臂），显存够吗？如果不够，有哪些办法？
2.  仿真里训到 90% 成功率的模型，到真机上只有 30%。你会先怀疑什么？按什么顺序排查？
3.  延迟预算里 VLA 推理占 25ms。能不能把 VLA 推理放到 L40 上、通过 WiFi 传结果回 NUC？这个方案有什么优缺点？

</div>

</div>
