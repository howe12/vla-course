# 第 2 章：仿真环境搭建

> MuJoCo + LIBERO 仿真平台，为 VLA 实验准备沙盒。

---

> **🔄 从 §1 的 VLA 理论出发** 你已经理解了 VLA 的三模态架构和动作离散化原理。现在在把这些概念落地的第一步------搭建仿真沙盒。MuJoCo 提供物理引擎，LIBERO 提供标准 VLA 评测任务，两者结合就是你的 VLA 实验室。

仿真环境搭建
------------

MuJoCo + LIBERO 仿真平台，为 VLA 实验准备沙盒。

> **🎯 学习目标**
>
> -   理解 MuJoCo 物理引擎的核心概念：mjModel / mjData / mj\_step
> -   掌握 LIBERO 基准测试：130 个标准化 VLA 任务的使用方法
> -   在本机搭建 MuJoCo + LIBERO 仿真环境（uv 管理依赖）
> -   跑通 LIBERO 随机动作 Demo，理解观测空间与动作空间
> -   为第 4 章 OpenVLA 推理准备好仿真沙盒

> **📍 本章定位**
>
> 本章是**VLA 实验的沙盒层**。所有后续章节的模型推理（§4 OpenVLA）、训练评估（§5 SmolVLA、§6 Pi0）都在 LIBERO 仿真中完成。装好这一章的环境，后面所有实验都能直接跑。

<div>

### 2.1 MuJoCo 物理引擎简介

**MuJoCo**（Multi-Joint dynamics with Contact）是 DeepMind 开发的高性能物理仿真引擎，专为机器人控制和强化学习设计。它是 LIBERO 的底层物理引擎------机械臂的关节运动、物体碰撞、摩擦力都由 MuJoCo 计算。

> **🔑 三个核心概念**
>
> -   **mjModel**：物理模型的"图纸"------关节定义、质量、惯量、几何形状。仿真中只读，不变。
> -   **mjData**：仿真运行时的"状态快照"------关节角度、速度、接触力。每一步仿真都会更新。
> -   **mj\_step**：推进一个物理步长（默认 0.002s），根据力和约束更新 mjData。

<div>

### 2.2 安装 MuJoCo 仿真环境

在本机用 **uv** 创建项目级虚拟环境，不污染系统 Python：

    cd /develop/vla-course/codes
    uv venv
    source .venv/bin/activate
    uv pip install mujoco

验证安装：

    python -c "import mujoco; print(mujoco.__version__)"

</div>

<div>

### 2.2.1 MuJoCo 入门实验 1：最简物理仿真

理解 MuJoCo 的核心循环------30 行代码看小球自由落体：

> **🧪 实验 1：小球自由落体**
>
> 1.  **运行**：`cd /develop/vla-course/codes && source .venv/bin/activate && python step2_sim/mujoco_minimal.py`
> 2.  **观察**：终端打印时间、球位置、速度，与自由落体公式对比
> 3.  **修改**：把 XML 中 `pos="0 0 1.5"` 改成 `pos="0 0 5.0"`，重跑看落地时间变化

展开查看核心代码


    model = mujoco.MjModel.from_xml_string(xml)  // XML -> "图纸"
    data = mujoco.MjData(model)                   // 运行时"快照"

    for step in range(200):
        mujoco.mj_step(model, data)               // 推进 0.002s
        ball_z = data.qpos[2]                     // 读取位置

> **学会**
>
> -   **mjModel** = 物理世界的"图纸"（编译后的 XML）
> -   **mjData** = 运行时的"状态快照"（位置 qpos、速度 qvel）
> -   **mj\_step** = 推进一个物理步长（0.002s），200 步 \~ 0.4s

#### 实验 1 渲染截图

MuJoCo 离屏渲染------小球从 1.5m 高处自由落体，4 帧捕捉下落过程：


![球高1.5m](https://raw.githubusercontent.com/howe12/vla-course/main/figs/mujoco_ball_000.png) [step 0 h=1.50m]{style="font-size:.7rem;color:var(--muted)"}


![球高1.43m](https://raw.githubusercontent.com/howe12/vla-course/main/figs/mujoco_ball_060.png) [step 60 h=1.43m]{style="font-size:.7rem;color:var(--muted)"}


![球高1.21m](https://raw.githubusercontent.com/howe12/vla-course/main/figs/mujoco_ball_120.png) [step 120 h=1.21m]{style="font-size:.7rem;color:var(--muted)"}


![球高0.85m](https://raw.githubusercontent.com/howe12/vla-course/main/figs/mujoco_ball_180.png) [step 180 h=0.85m]{style="font-size:.7rem;color:var(--muted)"}


![球落地](https://raw.githubusercontent.com/howe12/vla-course/main/figs/mujoco_ball_landed.png) [step 199 落地]{style="font-size:.7rem;color:var(--muted)"}

终端输出（点击展开）


    模型有 7 个位置自由度, 6 个速度自由度
    物理步长 dt=0.0020s  重力 g=(0.0,0.0,-9.81)

     Step      time    ball_z     ball_vz
    ---------------------------------------------
        0    0.0020    1.5000     -0.0196
       20    0.0420    1.4909     -0.4120
       40    0.0820    1.4662     -0.8044
       60    0.1220    1.4258     -1.1968
       80    0.1620    1.3697     -1.5892
      100    0.2020    1.2979     -1.9816
      120    0.2420    1.2104     -2.3740
      140    0.2820    1.1072     -2.7664
      160    0.3220    0.9883     -3.1588
      180    0.3620    0.8537     -3.5512

     核心公式: 1 次 mj_step = 物理时间推进 0.002s


<div>

### 2.2.2 MuJoCo 入门实验 2：关节控制

理解 `ctrl` `qpos` 的 PD 跟踪关系 VLA 如何驱动机械臂：

> **🧪 实验 2：两阶段关节控制**
>
> 1.  **运行**：`cd /develop/vla-course/codes && source .venv/bin/activate && python step2_sim/mujoco_arm_basics.py`
> 2.  **观察**：终端显示收拢姿态展开姿态的 ctrl/qpos 跟踪误差
> 3.  **修改**：改 `pose_reach` 数组的值，观察不同姿态的关节角

展开核心概念


    data.ctrl[0] = 0.8          // 设置目标("转到 0.8 rad")
    mujoco.mj_step(model, data) // PD 控制器自动跟踪
    actual = data.qpos[7]       // 物理仿真后的实际角度

> **学会**
>
> -   **data.ctrl** = VLA 输出的目标（你想要的位置）
> -   **data.qpos** = 物理仿真后的实际位置（PD 控制器跟踪结果）
> -   **VLA 推理周期**：每 40ms 输出 20 次 mj\_step 机械臂到位
> -   qpos\[7:13\] 对应 6 个关节，qpos\[0:7\] 是自由体（目标方块）

#### 实验 2 渲染截图

两阶段关节控制------收拢 vs 展开前伸，末端垂直位移 **0.68m**：


![收拢姿态](https://raw.githubusercontent.com/howe12/vla-course/main/figs/mujoco_arm_tucked.png) [收拢姿态（ctrl=\[0, -0.3, -0.5, 0, 0.3, 0\]）]{style="font-size:.8rem;font-weight:600"}\
[末端 Z=0.53m 机械臂紧凑立在基座上方]{style="font-size:.7rem;color:var(--muted)"}


![展开姿态](https://raw.githubusercontent.com/howe12/vla-course/main/figs/mujoco_arm_reach.png) [展开前伸（ctrl=\[0.2, -1.2, -1.5, 0, -0.5, 0\]）]{style="font-size:.8rem;font-weight:600"}\
[末端 Z=-0.01m 肩肘弯折，末端前伸至桌面]{style="font-size:.7rem;color:var(--muted)"}

> **关键观察**
>
> ctrl 变化（6 个数值）→ PD 控制器驱动关节 → qpos 跟踪 → 末端从 0.53m 下移到桌面高度（-0.01m）。这就是 VLA 控制机械臂的底层机制：**模型输出目标关节角 → MuJoCo 物理仿真 → 机械臂到位**。

终端输出（点击展开）


    阶段 1: ctrl  [ 0.  -0.3 -0.5  0.   0.3  0. ]
      步   0: max|ctrl-qpos| = 0.4981
      步  50: max|ctrl-qpos| = 0.1965
      步 100: max|ctrl-qpos| = 0.0717
      步 150: max|ctrl-qpos| = 0.0240
      qpos track: [ 0.018 -0.303 -0.494  0.     0.292  0.   ]

    阶段 2: ctrl  [ 0.2 -1.2 -1.5  0.  -0.5  0. ]
      步   0: max|ctrl-qpos| = 1.0042
      步  50: max|ctrl-qpos| = 0.4432
      步 100: max|ctrl-qpos| = 0.1852
      步 150: max|ctrl-qpos| = 0.1860
      qpos track: [ 0.015 -1.028 -1.382  0.    -0.434  0.   ]

    200 步 x 0.002s = 0.4s 足以完成大范围关节移动

> **实操路线图**
>
> 以上两个 MuJoCo 实验建立底层直觉 下面用课程机械臂做完整仿真（sim\_vla\_arm.py） 再进入 LIBERO 标准评测。三层递进：**原理 定制 标准化**。

<div>

### 2.2.5 动手：加载机械臂 + 模拟 VLA 控制

在装 LIBERO 之前，先用课程自带的 6-DOF 机械臂模型跑通一个完整的仿真闭环：

> **🧪 🧪 第一个仿真实验：看机械臂动起来**
>
> 1.  **确认 MuJoCo**：`python3 -c "import mujoco; print(mujoco.__version__)"`
> 2.  **启动图形仿真**：`cd /develop/vla-course/codes && uv run python step2_sim/sim_vla_arm.py`
> 3.  **无头模式**（SSH/服务器）：`uv run python step2_sim/sim_vla_arm.py --headless`
> 4.  **观察**：图形模式下，MuJoCo Viewer 弹出，机械臂按正弦轨迹摆动，末端逐渐靠近红色目标方块

这个脚本模拟了 VLA 的完整控制循环：

    相机图像 ──→ [VLA 模型] ──→ 目标关节角
        ↑                            ↓
        └──── 末端位置 ←── [MuJoCo PD 控制]

目前 VLA 输出用正弦轨迹 mock。第 4 章会把 mock 替换为 OpenVLA 7B 的真实推理------到那时，你看到的机械臂运动就是大模型"看懂场景后做出的决策"。


![初始姿态](https://raw.githubusercontent.com/howe12/vla-course/main/figs/sim_pose0_start.png) [① 初始姿态（竖直）]{style="font-size:.7rem;color:var(--muted)"}


![基座旋转](https://raw.githubusercontent.com/howe12/vla-course/main/figs/sim_pose1_rotate.png) [② 基座旋转 + 肩部下压]{style="font-size:.7rem;color:var(--muted)"}


![前伸](https://raw.githubusercontent.com/howe12/vla-course/main/figs/sim_pose2_reach.png) [③ 肩肘弯折前伸]{style="font-size:.7rem;color:var(--muted)"}


![接近目标](https://raw.githubusercontent.com/howe12/vla-course/main/figs/sim_pose3_target.png) [④ 末端接近目标方块]{style="font-size:.7rem;color:var(--muted)"}

> **📸 姿态说明**
>
> 以上 4 帧来自 `sim_vla_arm.py` 的实际运行截图。机械臂从初始竖直姿态（①）→ 基座旋转加肩部下压（②）→ 肘部弯折前伸（③）→ 末端到达目标方块附近（④），垂直位移约 70 cm。当前用正弦轨迹驱动------第 4 章替换为 OpenVLA 推理后，同样的机械臂将根据语言指令做出有目的性的动作。


<div>

### 2.2.8 LIBERO 入门实验 1：任务浏览器

探索 LIBERO 的 6 个子集 + 130 个任务 理解 VLA 考什么：

> **🧪 实验 3：LIBERO 任务浏览器**
>
> 1.  **运行**：`cd /develop/vla-course/codes && source .venv/bin/activate && python step2_sim/libero_task_explorer.py`
> 2.  **观察**：6 个子集概览 Spatial 10 个任务的语言指令 单个任务的完整属性
> 3.  **思考**：为什么 VLA 评测需要这么多任务？一个任务不够吗？
>
> > **学会**
> >
> > -   LIBERO 有 4 个评测子集（Spatial/Object/Goal/100）+ 2 个训练子集
> > -   每个任务 = BDDL 场景描述 + 自然语言指令
> > -   观测空间包含 31 个 key：1 个图像 + 30 个本体/物体状态
> > -   动作空间 = 8 维（7 维末端增量 + gripper）
>
> <div>
>
> ### 2.2.9 LIBERO 入门实验 2：动作空间实验
>
> 反复测试不同动作值，理解每维动作的含义和限幅机制：
>
> > **🧪 实验 4：动作空间实验**
> >
> > 1.  **运行**：`cd /develop/vla-course/codes && source .venv/bin/activate && python step2_sim/libero_action_test.py`
> > 2.  **观察**：5 个实验（前进/右移/下降/夹爪/限位）的末端位置变化
> > 3.  **修改**：改 action 数组的值，体会增量控制的含义
>
> 展开关键结论
>

>     // 动作 = 末端增量（不是绝对位置！）
>     action[0] = +0.3  // X 前移
>     action[1] = -0.3  // Y 右移
>     action[6] = -1.0  // 夹爪关闭
>     action[7] =  0.0  // 保留位
>
>     // 即使 action=2.0（超大），实际位移也只有 ~0.06m
>     // 因为环境内部有动作限幅（clipping）
>
> > **学会**
> >
> > -   动作是末端增量，不是绝对位置
> > -   gripper\_ctrl：正=开，负=关
> > -   环境有内置的动作限幅保护
> > -   第 3 章 VLA 模型输出的就是这 7 个增量值
>
> #### 实验 4 运行结果
>
> 以下截图来自实验 4 的实际运行。每张图重复执行 20 步定向动作后拍摄：
>


> ![前移](https://raw.githubusercontent.com/howe12/vla-course/main/figs/action_forward.png) [X=+0.3 前移]{style="font-size:.7rem;color:var(--muted)"}
>

> ![右移](https://raw.githubusercontent.com/howe12/vla-course/main/figs/action_right.png) [Y=-0.3 右移]{style="font-size:.7rem;color:var(--muted)"}
>

> ![下降](https://raw.githubusercontent.com/howe12/vla-course/main/figs/action_down.png) [Z=-0.3 下降]{style="font-size:.7rem;color:var(--muted)"}
>


> ![夹爪开](https://raw.githubusercontent.com/howe12/vla-course/main/figs/action_gripper_open.png) [gripper=+1.0 夹爪打开]{style="font-size:.7rem;color:var(--muted)"}
>

> ![夹爪关](https://raw.githubusercontent.com/howe12/vla-course/main/figs/action_gripper_close.png) [gripper=-1.0 夹爪关闭（手指并拢）]{style="font-size:.7rem;color:var(--muted)"}
>
> > **夹爪对比**
> >
> > 夹爪开 vs 关是最容易看出的区别------观察机械臂末端的两根灰色手指：**打开时手指分开**（左图），**关闭时手指并拢**（右图）。第 4 章 OpenVLA 推理时，gripper 值会在接近物体时自动从 +1 变为 -1，实现"看到碗 → 伸手 → 抓住"的完整动作序列。
>
> > **实操路线图**
> >
> > MuJoCo 底层（实验 1-2） LIBERO 标准评测（实验 3-4） 课程机械臂实战（下方 2.2.5） LIBERO Demo（2.4.5）。四层递进：**原理 探索 定制 标准化**。
>
> <div>
>
> ### 2.3 LIBERO 基准测试
>
> **LIBERO** 是 Stanford 等机构联合发布的机器人操作基准，包含 130 个标准化任务，分为 4 个子集：
>
>   子集                 任务数   考察能力
>   -------------------- -------- -------------------------------------
>   **LIBERO-Spatial**   10       空间关系理解（把 A 放到 B 左边）
>   **LIBERO-Object**    10       物体识别（挑出特定颜色/形状的物体）
>   **LIBERO-Goal**      10       目标导向（按指令完成多步操作）
>   **LIBERO-100**       100      综合评测（前三个的超集 + 更多任务）
>
> > **🔑 为什么学 LIBERO？**
> >
> > OpenVLA 官方在 LIBERO 上评测，SmolVLA 论文也用它做对比实验。学会 LIBERO = 能和 SOTA 论文的表格直接对齐。你的模型在 LIBERO-Spatial 上跑到 80%，就能和 OpenVLA 原论文的数字比。
>
> </div>
>
> <div>
>
> ### 2.4 动手实操：跑通 LIBERO
>
> <div>
>
> #### 2.4.1 安装 LIBERO
>
>     cd /develop/vla-course/codes
>     source .venv/bin/activate
>     # 国内需走代理下载 HuggingFace 资产
>     export http_proxy=http://127.0.0.1:7897
>     uv pip install libero
>
> </div>
>
> <div>
>
> #### 2.4.5 动手：跑通 LIBERO 随机动作 Demo
>
> 运行 `codes/step2_sim/run_libero_demo.py`（80 步随机动作），下面是实际运行中截取的 **agentview 观测帧**：
>
> > **🧪 🧪 LIBERO 仿真验证实验**
> >
> > 1.  **进入目录**：`cd /develop/vla-course/codes`
> > 2.  **激活环境**：`source .venv/bin/activate`
> > 3.  **运行**：`python step2_sim/run_libero_demo.py`
> > 4.  **查看输出**：脚本自动保存 5 帧观测截图到 `step2_sim/_screenshots/`
>


> ![LIBERO Step 0](https://raw.githubusercontent.com/howe12/vla-course/main/figs/libero_step_000.png) [① 初始悬停（默认姿态）]{style="font-size:.7rem;color:var(--muted)"}
>

> ![LIBERO Step 20](https://raw.githubusercontent.com/howe12/vla-course/main/figs/libero_step_020.png) [② 右移下降（靠近桌面）]{style="font-size:.7rem;color:var(--muted)"}
>

> ![LIBERO Step 40](https://raw.githubusercontent.com/howe12/vla-course/main/figs/libero_step_040.png) [③ 大幅左移（横跨桌面）]{style="font-size:.7rem;color:var(--muted)"}
>

> ![LIBERO Step 60](https://raw.githubusercontent.com/howe12/vla-course/main/figs/libero_step_060.png) [④ 前伸 + 夹爪闭合]{style="font-size:.7rem;color:var(--muted)"}
>

> ![LIBERO Step 80](https://raw.githubusercontent.com/howe12/vla-course/main/figs/libero_step_080.png) [⑤ 上提取物（模拟举起）]{style="font-size:.7rem;color:var(--muted)"}
>
> > **📸 观测帧说明**
> >
> > 以上 5 帧来自 **pick up the black bowl between the plate and the ramekin and place it on the plate** 任务。脚本用**大幅定向动作**（而非随机抖动）驱动 Panda 机械臂完成一个模拟抓取序列：① 初始悬停 → ② 右移下降靠近桌面 → ③ 大幅左移横跨桌面 → ④ 前伸 + 夹爪闭合（模拟抓取）→ ⑤ 上提取物（模拟举起）。
> >
> > **💡 顶视角局限**：LIBERO 默认只有顶部固定相机（agentview），机械臂在画面中只占 \~12% 像素，因此缩略图看起来差异不大。**请点击大图仔细观察机械臂的 XY 位置和夹爪开合状态。**第 4 章用 OpenVLA 替换定向动作后，机械臂将根据语言指令自主完成真实抓取。

>
> <div>
>
> ### 2.5 第一个 VLA 推理 Demo
>
> 安装完环境后，跑通一个最小化的 VLA 推理流程------用 OpenVLA 在 LIBERO-Spatial 上做一个零样本推理：
>
> > **🧪 🧪 跑通第一个 VLA 仿真实验**
> >
> > 1.  **确认环境**：`cd /develop/vla-course/codes && uv run python -c "import mujoco, libero; print('OK')"`
> > 2.  **创建脚本**：在 `codes/step2_sim/` 下新建 `run_openvla.py`（见下方折叠面板）
> > 3.  **运行**：`uv run python codes/step2_sim/run_openvla.py`
> > 4.  **观察**：终端打印每个 step 的预测动作和 gripper 状态
>
> 展开查看 run\_openvla.py 完整代码
>

>     #!/usr/bin/env python3
>     """"最小化 VLA 推理：OpenVLA + LIBERO-Spatial""""
>     import numpy as np
>     from libero.libero import benchmark
>
>     # 1. 加载 LIBERO 任务
>     print("加载 LIBERO-Spatial...")
>     benchmark_dict = benchmark.get_benchmark_dict()
>     task_suite = benchmark_dict["libero_spatial"]()
>     task = task_suite.get_task(0)
>
>     print(f"任务: {task.name}")
>     print(f"指令: {task.language_instruction}")
>
>     # 2. 创建环境
>     env_args = {
>         "task": task,
>         "bddl_file": task.bddl_file,
>         "camera_heights": 224,
>         "camera_widths": 224,
>     }
>     env = task_suite.get_env(task_id=0, env_args=env_args)
>
>     # 3. 重置 → 推理循环
>     obs = env.reset()
>     print(f"观测 shape: {obs.shape}  (应为 224,224,3)")
>     print(f"动作维度: {env.action_space.shape[0]}  (应为 7)")
>
>     # 模拟 VLA 推理（实际模型加载见第 4 章）
>     for step in range(20):
>         # ── VLA 推理位置（第 4 章替换为真实模型）──
>         action = np.random.uniform(-0.1, 0.1, 7)
>         action[6] = 1.0  # gripper 保持张开
>
>         obs, reward, done, info = env.step(action)
>
>         if step % 5 == 0:
>             print(f"Step {step:3d} | 动作 Δxyz=({action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f}) | 夹爪={action[6]:.1f} | done={done}")
>
>         if done:
>             print(f"✅ 任务完成于 step {step}")
>             break
>
>     print("仿真结束。下一步：第 4 章加载真实 OpenVLA 模型替换随机动作。")
>
> 这个脚本目前用随机动作占位。第 3 章会加载真实的 OpenVLA 7B 模型，把 `np.random.uniform` 替换为模型推理输出。

>
> <div>
>
> ### 📝 本章小结
>
> > **带走这几条**
> >
> > 1.  **MuJoCo = 物理根基**：mjModel 是图纸，mjData 是实时状态，mj\_step 推进仿真
> > 2.  **LIBERO = VLA 考场**：130 个标准任务，gym.Env 接口，OpenVLA 原生支持
> > 3.  **动作空间 = 7 维增量**：Δx/Δy/Δz/Δroll/Δpitch/Δyaw + gripper，和 §1.5 的 256 桶离散化一一对应
> > 4.  **控制周期 = 20 Hz**：VLA 推理 40ms → 25Hz，LIBERO 默认 20Hz 控制，二者匹配
> > 5.  **uv 管理依赖**：项目级隔离，不污染系统 Python，不冲突 conda
>
> #### 🤔 思考题
>
> 1.  MuJoCo 的物理步长默认 0.002s，LIBERO 的控制周期是 0.05s（20 Hz）。这一个控制周期内 mj\_step 被调用了多少次？为什么要这样设计？
> 2.  LIBERO 的动作空间是末端增量而非绝对位置------这个设计对 VLA 模型有什么好处？（提示：结合 §1.8 的连续 vs 离散动作讨论）
> 3.  为什么 VLA 评测需要 LIBERO-100 这样的 100 任务套件，而不能只在一个任务上测试？这反映了 VLA 的什么核心挑战？
>
> </div>
>
> </div>





>
> </div>
>
> </div>








>
> </div>

</div>


</div>


</div>


</div>

</div>
