# VLA 从零到部署 — 具身智能教科书

> 具身智能 · MuJoCo + LeRobot + L40 + Gemini

---

# 第 1 章：VLA 基础与架构

> 视觉-语言-动作模型的核心概念、技术演进与系统架构。

---

> **🔄 从 VLN 和模仿学习出发** 你已经让机器人在 Habitat 里听懂指令导航（VLN），也让机械臂复现过人类的演示轨迹（模仿学习）。现在把这两条线**拧成一股绳**------用一个统一的大模型同时理解语言、看懂图像、输出动作。

VLA 基础与架构
--------------

视觉-语言-动作模型的核心概念、技术演进与系统架构。

> **🎯 学习目标**
>
> -   理解 VLA 的定义、数学形式，与 VLN / VLM / 模仿学习的本质区别
> -   掌握 VLA 三模态架构：视觉编码器 → LLM 主干 → 动作预测头
> -   理解动作离散化的原理（256 桶方案）及其与 LLM Token 的统一
> -   了解 RT-1→RT-2→OpenVLA→π0 的技术演进和设计取舍
> -   清楚连续动作 vs 离散动作的工程权衡，以及实时推理的优化手段

> **📍 本章定位**
>
> 本章是 VLA 课程的**理论基础**。你已经完成了 VLN（视觉语言导航）和模仿学习（ACT/LeRobot），VLA 是这两条线的交汇------用一个统一的大模型同时理解语言、看懂图像、输出动作。第 3-5 章的 OpenVLA / SmolVLA / Pi0 实战，都建立在本章的概念之上。

<div>

### 1.1 什么是 VLA？

**视觉-语言-动作模型（Vision-Language-Action Model，VLA）** 是一类将视觉感知、语言理解和机器人动作控制统一集成到同一个神经网络中的端到端模型。与 VLM 仅输出文本不同，VLA 的核心目标是**根据视觉观察和语言指令，直接输出机器人可执行的动作序列**，实现"所见即所动"的具身智能控制。

> **🔑 一句话定义**
>
> ##### VLA：图像 + 语言指令 → 动作序列
>
> VLM 解决"感知-理解"问题（让机器"看懂"），VLA 解决"感知-理解-执行"问题（让机器"看懂并动手"）。VLA 可以看作在 VLM 的能力基础上，增加了**动作预测**这一关键能力。


p(**a**~1:T~ \| **X**~img~, **X**~text~) = ∏~t=1~^T^ p(**a**~t~ \| **X**~img~, **X**~text~, **a**~1:t-1~)

其中 **a**~t~ 为 t 时刻机器人执行的动作，**X**~img~ 为视觉观察，**X**~text~ 为语言指令。这是一个自回归生成过程------每一步的动作预测都依赖于之前已经执行的动作历史。


<div>

### 1.2 VLA vs VLM vs VLN vs 模仿学习

四种让机器人"听懂人话"的范式，本质区别在于**输出模态**和**泛化方式**：

  范式           输入             输出           核心思路               局限
  -------------- ---------------- -------------- ---------------------- --------------------------------
  **VLM**        图像+文本        文本           描述/问答/理解场景     不能控制机器人
  **VLN**        全景图+指令      离散导航动作   专用模型，只做导航     任务单一，无法操作
  **模仿学习**   图像+本体状态    连续动作       从人类演示中学习策略   每任务需新数据，不利用语言先验
  **VLA**        图像+语言+状态   动作序列       统一大模型，端到端     训练成本高，实时性挑战

> **🔑 动作空间的含义**
>
> LIBERO 使用的是**末端执行器增量控制**：每一步预测的是末端位姿的**变化量**（而非绝对位置）。这和 OpenVLA 原论文的动作表示完全一致------§1.5 讲的 256 桶离散化，离散化的就是这 7 个维度的增量值。

</div>

<div>

### 1.9 动手实验 1：动作离散化

不加载任何模型，用 30 行 Python 亲手把连续动作映射到 256 个离散桶：

> **🧪 实验 1.1：动作离散化（256 桶方案）**
>
> 1.  **运行**：`cd /develop/vla-course/codes && source .venv/bin/activate && python step1_basics/action_discretize.py`
> 2.  **观察**：7 维动作 256 桶索引 恢复值，误差 \< 0.008
> 3.  **修改**：改 NUM\_BINS=128 或 512，观察精度变化

展开核心输出


    动作维度: 7  (x, y, z, roll, pitch, yaw, gripper)
    值域: [-1.0, 1.0]
    桶数: 256
    分辨率: 0.00784

          维度       连续值     桶索引       恢复值          误差
    -------------------------------------------------------
          x    0.3500     172    0.3490     0.00098
          y   -0.1200     112   -0.1216     0.00157
          z    0.0800     137    0.0745     0.00549
       roll    0.0000     127   -0.0039     0.00392
      pitch   -0.0500     121   -0.0510     0.00098
        yaw    0.0200     130    0.0196     0.00039
     gripper    1.0000     255    1.0000     0.00000

> **学会**
>
> -   连续值 离散桶的公式: `bucket = (val+1)/2 255`
> -   离散化精度 = 2/255 0.0078，对机械臂控制足够
> -   256 = 2 1 个字节刚好装下 1 个 token ID


<div>

### 1.10 动手实验 2：Token 编码解码

理解 OpenVLA 如何把 7 维动作编成 8 个 token ID，LLM 自回归生成后解码回动作：

> **🧪 实验 1.2：动作 Token ID 编码解码**
>
> 1.  **运行**：`cd /develop/vla-course/codes && source .venv/bin/activate && python step1_basics/token_encode_decode.py`
> 2.  **观察**：7 维动作 1792 个 token ID LLM 生成 解码回动作
> 3.  **修改**：改 action 数组的值，观察 token ID 如何变化

展开核心输出


    输入动作: [0.35, -0.12, 0.78, -0.05, 0.0, 0.02, 0.5]

          维度       连续值     桶索引    Token ID   ID//256    ID%256
    -----------------------------------------------------------------
          x    0.3500     172         172         0       172
          y   -0.1200     112         368         1       112
          z    0.7800     226         738         2       226
       roll   -0.0500     121         889         3       121
      pitch    0.0000     127        1151         4       127
        yaw    0.0200     130        1410         5       130
     gripper    0.5000     191        1727         6       191

    Token ID = dim  256 + bucket
      解码: dim = ID // 256, bucket = ID % 256
      恢复误差 < 0.008

> **学会**
>
> -   **Token ID = dim 256 + bucket** --- 7 维动作编成 7 个整数
> -   **ID // 256 = 维度，ID % 256 = 桶** --- 数学上可逆，无信息丢失（精度损失除外）
> -   LLaMA 词汇表从 \~32000 扩展到 \~33792，加入 1792 个动作 token
> -   第 8 个 token 是停止符（EOS），告诉 LLM 动作输出完毕


<div>

### 📝 本章小结

> **带走这几条**
>
> 1.  **VLA = 视觉 + 语言 + 动作**的统一端到端模型，区别于 VLM（只输出文本）和 VLN（只做导航）
> 2.  **三模态架构**：视觉编码器 → LLM 主干 → 动作预测头，是当前主流设计
> 3.  **动作离散化**：256 桶方案让连续动作和 LLM Token 统一，自回归生成
> 4.  **技术演进**：RT-1→RT-2→OpenVLA→π0，从闭源到开源、从巨型到可训
> 5.  **连续 vs 离散**：离散化方便复用 LLM 架构，连续动作需要专门的扩散/流匹配方案

#### 🤔 思考题

1.  如果把 VLA 的动作空间从 7 维（末端增量）扩展到全身人形机器人的 50+ 自由度，离散化的 256 桶方案还 work 吗？会遇到什么瓶颈？
2.  从 RT-1 到 π0，视觉编码器从 EfficientNet 换成了 ViT 又换成了扩散模型------为什么视觉编码器的设计对 VLA 如此关键？
3.  一个 VLA 模型在仿真中达到 90% 成功率，部署到真机能保持多少？可能的 gap 来源有哪些？

</div>

</div>

</div>

</div>


---

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


---

# 第 3 章：L40 云端训练环境

> SSH 连接、CUDA 配置、依赖管理——为后续训练备好算力。

---

> **🔄 从 §2 的仿真环境出发** MuJoCo + LIBERO 已就绪。现在把**云端算力**搭好------L40 GPU 是后续所有训练的硬件基础。本章配置 SSH、CUDA、tmux，让 SmolVLA（§5）和 Pi0（§6）的训练"开箱即用"。

L40 云端训练环境
----------------

SSH 连接、CUDA 配置、依赖管理------为后续训练备好算力。

> **🎯 学习目标**
>
> -   SSH 免密登录 L40 云端 GPU 服务器
> -   理解 L40 的硬件配置和适用场景（44GB 显存 / 海外机房）
> -   安装 CUDA Toolkit + PyTorch + 核心依赖
> -   掌握 scp/rsync 文件传输和 tmux 持久会话
> -   跑通 GPU 验证脚本，确认环境可用于后续训练

> **📍 本章定位**
>
> 本章是**算力基础设施**。§5 的 SmolVLA 训练（450M, \~6h）和 §6 的 Pi0 训练（3B, \~20h）都需要 GPU。L40 的 44GB 显存可以同时覆盖这两个模型------装好一次环境，后面两章都不用再碰配置。

<div>

### 3.1 为什么选 L40？

训练 VLA 模型对 GPU 有硬性要求：OpenVLA 7B 的 LoRA 微调需要 20-30GB，SmolVLA 全量训练需要 16-24GB，Pi0 需要 30-40GB。消费级显卡（RTX 3090 24GB / 4060 8GB）只能做推理，训练捉襟见肘。

  GPU        显存    SmolVLA 训练       Pi0 训练            价格
  ---------- ------- ------------------ ------------------- ------------
  **L40**    44 GB   ✅ batch=64, \~6h   ✅ batch=16, \~20h   \~\$0.80/h
  A100       80 GB   ✅ batch=128        ✅ batch=64          \~\$2.50/h
  RTX 3090   24 GB   ⚠️ batch=8         ❌ OOM               自有硬件
  RTX 4060   8 GB    ❌ OOM              ❌ OOM               自有硬件

> **🔑 L40 的三个关键优势**
>
> 1.  **44GB 显存**：正好卡在"够用"的甜点------SmolVLA 可以开大 batch，Pi0 刚好能跑
> 2.  **海外机房**：直连 HuggingFace / GitHub，不用配镜像、不用翻墙，模型下载满速
> 3.  **性价比**：\$0.80/h 是 A100 的三分之一，训一晚上不到 \$10

</div>

<div>

### 3.2 SSH 连接与免密登录

L40 的 SSH 连接信息（已记录在你的记忆中）：

    # SSH 连接
    ssh -p 30334 root@120.209.70.195

    # 设置免密登录（本机执行一次）
    ssh-copy-id -p 30334 root@120.209.70.195

    # 之后直接连，无需密码
    ssh -p 30334 root@120.209.70.195

#### 推荐：配置 SSH config 别名

    # 在本机 ~/.ssh/config 中添加
    Host l40
        HostName 120.209.70.195
        Port 30334
        User root
        IdentityFile ~/.ssh/id_rsa

    # 之后只需
    ssh l40

> **⚠️ ⚠️ L40 是手动开关机**
>
> L40 通过云控制台手动开关机，不使用时记得关机（按小时计费）。开机后需要等 1-2 分钟系统启动。如果 SSH 连不上，先去云控制台确认机器状态。

</div>

<div>

### 3.3 硬件与环境概览

登录 L40 后，先确认硬件配置：

    # 查看 GPU
    nvidia-smi
    # 应输出: NVIDIA L40, 46068 MiB

    # 查看 CUDA 版本
    nvcc --version
    # 应输出: CUDA 12.x

    # 查看磁盘
    # 工程路径: /root/gpufree-data/habitat-lab-edu/（PVC 持久化）
    df -h /root/gpufree-data

L40 的核心硬件配置：

  硬件       规格
  ---------- ------------------------------------------
  **GPU**    NVIDIA L40, 48GB 显存, Ada Lovelace 架构
  **CUDA**   12.x
  **CPU**    Intel Xeon, \~16 vCPU
  **RAM**    60GB+
  **磁盘**   PVC 持久化存储, /root/gpufree-data/

> **📍 PVC 持久化存储**
>
> L40 的 `/root/gpufree-data/` 是 PVC 持久化卷------**关机不会丢数据**。训练 checkpoints、下载的模型权重、实验代码都应该放在这个目录下。但注意：**Git 提交还是要 push**------PVC 是本机的持久化，不是远程备份。

</div>

<div>

### 3.4 Python 环境配置

L40 上已有 Python 3.9 环境和 CUDA。本章用 **uv** 管理项目依赖，与本机 §2 的配置保持一致：

    # L40 上执行（首次配置）
    # 1. 确认 Python
    python3 --version  # Python 3.9.x

    # 2. 安装 uv（如果没有）
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env

    # 3. 创建 vla-course 工程目录
    mkdir -p /root/gpufree-data/vla-course
    cd /root/gpufree-data/vla-course

    # 4. 初始化 uv 项目
    uv init

    # 5. 安装 PyTorch（CUDA 12.x）
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # 6. 安装 VLA 训练核心依赖
    uv pip install transformers accelerate datasets
    uv pip install peft bitsandbytes  # LoRA + 量化

#### 验证 GPU 可用

    uv run python -c "
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    "
    # 预期输出:
    # PyTorch: 2.x.x
    # CUDA available: True
    # GPU: NVIDIA L40
    # VRAM: 44.3 GB

</div>

<div>

### 3.5 文件传输：本机 ↔ L40

课程代码在本机开发，训练在 L40 上跑。需要可靠的文件同步方案：

    # 方式 1：scp（单文件/小目录）
    # 本机 → L40
    scp -P 30334 /develop/vla-course/codes/step5_smolvla/train.py root@120.209.70.195:/root/gpufree-data/vla-course/

    # L40 → 本机（下载 checkpoint）
    scp -P 30334 root@120.209.70.195:/root/gpufree-data/vla-course/ckpt/best.pt /develop/vla-course/ckpt/

    # 方式 2：rsync（大目录、增量同步、推荐）
    # 本机 → L40（只传变化的文件）
    rsync -avz -e "ssh -p 30334" \
      /develop/vla-course/codes/ \
      root@120.209.70.195:/root/gpufree-data/vla-course/codes/

    # L40 → 本机（下载训练产物）
    rsync -avz -e "ssh -p 30334" \
      root@120.209.70.195:/root/gpufree-data/vla-course/outputs/ \
      /develop/vla-course/outputs/

> **🔑 scp vs rsync**
>
> **scp** 适合传单个文件（模型权重、checkpoint）。**rsync** 适合同步整个目录（代码、数据集），只传变化的文件，中断后可续传。课程后续的"本机编辑 → rsync 推送 → L40 训练 → rsync 拉回"工作流基于 rsync。

</div>

<div>

### 3.6 tmux：让训练在后台跑

训练 SmolVLA 需要 \~6 小时，Pi0 需要 \~20 小时。SSH 断开会导致前台进程被杀。**tmux** 让训练会话在后台持久运行：

    # 创建新会话
    ssh l40
    tmux new -s train

    # 在 tmux 里启动训练
    cd /root/gpufree-data/vla-course
    uv run python codes/step5_smolvla/train.py

    # 断开 tmux（训练继续在后台跑）
    Ctrl+B, D

    # 重新连接
    tmux attach -t train

    # 查看所有会话
    tmux ls

#### tmux 常用快捷键

  操作       快捷键
  ---------- ------------------------
  断开会话   `Ctrl+B, D`
  新建窗口   `Ctrl+B, C`
  切换窗口   `Ctrl+B, 0-9`
  上下滚动   `Ctrl+B, [` 然后方向键
  水平分屏   `Ctrl+B, "`
  垂直分屏   `Ctrl+B, %`

</div>

<div>

### 3.7 GPU 监控：实时看显存和利用率

训练中需要监控 GPU 状态------显存是否 OOM、利用率是否打满：

    # 基础监控（实时刷新）
    watch -n 1 nvidia-smi

    # 更详细的监控（推荐安装 gpustat）
    uv pip install gpustat
    gpustat -i 1  # 每秒刷新

    # 输出示例:
    # [0] NVIDIA L40 | 48°C,  85% utilization,  32768 / 46068 MB
    #     pid 12345: python train.py  (32600 MB)


⚠️ 训练中关注

-   **显存 (Memory)**：接近 44GB 红线时减小 batch\_size
-   **利用率 (Utilization)**：应该 \> 80%，如果 \< 50% 可能是数据加载瓶颈
-   **温度 (Temp)**：L40 正常在 50-70°C，超过 80°C 检查散热


<div>

### 3.8 动手实验：初始化 L40 训练环境

> **🧪 🧪 把 L40 环境从头配好**
>
> 1.  **SSH 登录**：`ssh -p 30334 root@120.209.70.195`
> 2.  **创建工程目录**：`mkdir -p /root/gpufree-data/vla-course && cd /root/gpufree-data/vla-course`
> 3.  **初始化 uv**：`uv init && uv pip install torch --index-url https://download.pytorch.org/whl/cu121`
> 4.  **验证 GPU**：`uv run python -c "import torch; print(torch.cuda.get_device_name(0))"`
> 5.  **启动 tmux**：`tmux new -s test`
> 6.  **运行 GPU 压测**：`uv run python codes/step4_l40/gpu_test.py`

展开查看 gpu\_test.py（GPU 基准测试脚本）


    #!/usr/bin/env python3
    """L40 GPU 基准测试 - 验证环境可用、测试算力上限"""
    import torch
    import time

    print("=" * 50)
    print("L40 GPU 基准测试")
    print("=" * 50)

    # 1. 基本信息
    print(f"\
    PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_mem / 1024**3:.1f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"CUDA Cores: {props.multi_processor_count} SM")

    # 2. 矩阵乘法算力测试
    print("\
    --- 矩阵乘法算力测试 ---")
    for size in [1024, 2048, 4096, 8192]:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        
        # 预热
        for _ in range(5):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        for _ in range(20):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tflops = 2 * size**3 * 20 / elapsed / 1e12
        print(f"  {size}x{size} matmul: {elapsed/20*1000:.1f}ms ({tflops:.1f} TFLOPS)")

    # 3. 可用显存测试
    print("\
    --- 显存测试 ---")
    for frac in [0.25, 0.5, 0.75, 0.9]:
        try:
            total = torch.cuda.get_device_properties(0).total_mem
            size = int(total * frac) // 4  # float32
            x = torch.zeros(size, device="cuda")
            used = torch.cuda.memory_allocated() / 1024**3
            print(f"  分配 {frac*100:.0f}% 显存: {used:.1f} GB OK")
            del x
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  分配 {frac*100:.0f}% 显存: OOM")

    print("\
    GPU 基准测试完成！")
    print("L40 环境已就绪，可以开始下一步训练。")

跑通这个实验后，L40 环境就完整配置好了。§5 的 SmolVLA 训练脚本直接 scp 上去就能跑。


<div>

### 📝 本章小结

> **带走这几条**
>
> 1.  **L40 的 44GB 显存**恰好覆盖 SmolVLA (16-24GB) 和 Pi0 (30-40GB) 的训练需求
> 2.  **SSH 免密 + config 别名**：一行 `ssh l40` 连接，不输密码
> 3.  **PVC 持久化**：`/root/gpufree-data/` 关机不丢数据，checkpoint 放这里
> 4.  **uv 统一管理**：本机和 L40 用同一套依赖方案，避免版本差异
> 5.  **rsync + tmux**：代码推送 → tmux 后台训练 → 断开 SSH → 过几小时回来收结果

#### 🤔 思考题

1.  如果把 gpu\_test.py 在 RTX 4060 (8GB) 上跑，哪个显存分配比例会 OOM？
2.  rsync 和 scp 都可以传文件------如果一个 10GB 的模型权重已传到 L40 上，你又微调了一下训练脚本（5KB），用什么命令只更新脚本？
3.  tmux 断开后训练继续跑------但如果你重启了 L40（关机再开机），tmux 会话还在吗？怎么办？

> **🔄 从 §3 的 L40 云端环境出发** GPU 已就绪、tmux 已学会。现在进入 VLA 推理实战------加载 OpenVLA 7B，让大模型看懂场景、理解指令、输出可以完成任务的机械臂动作。这也是你第一次看到 VLA 模型真正控制机器人。

L40 云端训练环境
----------------

SSH 连接、CUDA 配置、依赖管理------为后续训练备好算力。

> **🎯 学习目标**
>
> -   理解 OpenVLA 的架构：SigLIP ViT + LLaMA 7B + Action Head 的分工
> -   从 HuggingFace 下载并加载 OpenVLA 7B 模型（FP16 / INT4）
> -   掌握推理管线：图像预处理 → 多模态前向 → 离散 Token → 连续动作
> -   掌握推理管线：图像预处理 模型前向 动作解码 环境闭环
> -   在 LIBERO-Spatial 上运行完整的 VLA 微调模型推理闭环

> **📍 本章定位**
>
> 本章是**VLA 推理的核心章**。你先学会"用"模型------加载现成的 OpenVLA 权重，理解它如何从像素走到关节角。第 5 章（SmolVLA）和第 6 章（Pi0）会教你"训"模型。

<div>

### 4.1 OpenVLA 是什么？

**OpenVLA**（2024.3）是 Stanford HAI 和 UIUC 联合发布的**首个开源 7B VLA 模型**。它的发布标志着 VLA 从"闭源封闭"走向"开源开放"------权重、代码、推理框架全部公开，Apache 2.0 协议可商用。

  特性             OpenVLA                        RT-2（对比）
  ---------------- ------------------------------ ---------------------------------
  **参数量**       7B                             55B
  **视觉编码器**   SigLIP ViT-SO (384M)           ViT-22B
  **语言基座**     LLaMA 2 7B                     T5-XL + PaLI-X
  **训练数据**     Open X-Embodiment (2M+ 轨迹)   内部数据 (13 台机器人, 17 个月)
  **推理速度**     \~40ms (INT4, 25Hz)            \~530ms (FP16, \~2Hz)
  **开源**         ✅ Apache 2.0                   ❌ 闭源

> **🔑 核心创新**
>
> OpenVLA 证明了**开源 LLM（LLaMA 2）可以直接做 VLA 的基座**，不需要 Google 的闭源模型。7B 参数经过 INT4 量化后仅需 \~5GB 显存，可以在消费级 GPU（RTX 3090/4060）上推理。这也是本课程选它的原因------**你能在自己的硬件上跑起来**。

</div>

<div>

### 4.2 OpenVLA 架构详解

OpenVLA 遵循"视觉编码器 → 语言基座 → 动作头"三阶段范式，与 §1 讲的理论架构完全对应：

    ┌──────────────────────────────────────────────────────┐
    │  📷 RGB 图像 (224×224×3)                              │
    │        ↓                                              │
    │  SigLIP ViT-SO (384M)                                │
    │  将图像切为 14×14 = 196 个 patch，输出 196×1024 特征   │
    │        ↓                                              │
    │  ┌─────────────────────────────────────────┐         │
    │  │  Projector: 1024 → 4096 (LLaMA 维度)    │         │
    │  └─────────────────────────────────────────┘         │
    │        ↓                                              │
    │  ┌─────────────────────────────────────────┐         │
    │  │  LLaMA 2 7B (32 层, 4096 hidden)        │         │
    │  │  · 196 个视觉 token + N 个语言 token    │         │
    │  │  · 自回归生成 8 个动作 token             │         │
    │  └─────────────────────────────────────────┘         │
    │        ↓                                              │
    │  ┌─────────────────────────────────────────┐         │
    │  │  Action Head: 每个 token → 256 类分类   │         │
    │  │  7 个动作维 × 256 桶 → 7 个离散索引     │         │
    │  └─────────────────────────────────────────┘         │
    │        ↓                                              │
    │  🦾 连续动作: (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, g)  │
    └──────────────────────────────────────────────────────┘

#### 三个组件的分工

  组件             模型                        参数量   作用
  ---------------- --------------------------- -------- --------------------------------------
  **视觉编码器**   SigLIP ViT-SO               \~384M   RGB → 196 个 1024 维视觉 token
  **投影层**       Linear(1024→4096)           \~4M     视觉维度 → LLaMA 维度对齐
  **语言基座**     LLaMA 2 7B                  \~6.7B   融合视觉+语言 → 自回归生成动作 token
  **动作头**       7 个独立 Linear(4096→256)   \~7M     每个 token → 256 桶概率 → 连续动作值

</div>

<div>

### 4.3 模型下载与加载

OpenVLA 模型托管在 HuggingFace，约 14GB（FP16）。国内下载可能需要镜像：

    # 方案 A：直接下载（推荐，国外网络或代理）
    pip install transformers>=4.40 torch>=2.0 accelerate

    # 方案 B：HF 镜像（国内网络）
    export HF_ENDPOINT=https://hf-mirror.com
    pip install transformers torch accelerate

    # 方案 C：L40 云端（第 3 章）
    # L40 位于海外机房，直连 HuggingFace，无需镜像

#### 加载模型（FP16）

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_id = "openvla/openvla-7b"

    # FP16 加载（需 ~14GB 显存）
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",          # 自动分配 GPU/CPU
        trust_remote_code=True,     # OpenVLA 有自定义代码
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print("模型加载完成！")

#### INT4 量化加载（消费级 GPU）

    # INT4 量化（需 ~8GB 显存，RTX 4060 / 3090 可用）
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )


⚠️ 硬件要求

-   **FP16**：\~14GB 显存 --- RTX 3090/4090, A100, L40
-   **INT8**：\~8GB 显存 --- RTX 3070/4060, V100
-   **INT4**：\~5GB 显存 --- RTX 2060/3060, T4
-   **CPU**：理论上可行（\~64GB RAM），推理速度 \~2-5 秒/步，不推荐


<div>

### 4.4 推理管线：从像素到关节角

一次完整的 OpenVLA 推理包含四个步骤。理解每一步，你就能在任何机器人平台上适配它：

#### 步骤 1：图像预处理

    from PIL import Image
    import numpy as np

    # LIBERO 返回的 RGB 数组 (224, 224, 3) uint8
    obs = env.reset()  # numpy uint8
    image = Image.fromarray(obs).convert("RGB")

    # 可选：反归一化到 [0,1]（OpenVLA processor 会自动处理）

#### 步骤 2：构建 Prompt

    # OpenVLA 使用特殊格式编码任务指令
    # 格式: "Task: {instruction}. Bot: In: Im"
    instruction = task.language_instruction
    prompt = f"Task: {instruction}. Bot: In: Im"

    # processor 自动将 "Im" 替换为图像 token
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(model.device)

#### 步骤 3：模型前向推理

    with torch.no_grad():
        # 自回归生成 8 个 token（7 动作 + 1 结束符）
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,          # 贪心解码 = 确定性输出
            num_beams=1,
        )

    # 提取新生成的 token
    action_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    # shape: (1, 8)

#### 步骤 4：Token → 连续动作

    # OpenVLA 的 action head 是 7 个 Linear(4096, 256) 分类头
    # generate 的输出 token ID 就是每个维度的桶索引 (0~255)

    def decode_action(action_tokens, num_bins=256):
        """
        将离散 token 恢复为连续动作值
        
        action_tokens: (1, 8) — 最后一个是结束符，忽略
        返回: (7,) numpy array, 范围 [-1, 1]
        """
        token_ids = action_tokens[0, :7].cpu().numpy()  # 取前 7 个
        
        actions = []
        for tid in token_ids:
            # 256 桶均匀分布在 [-1, 1]
            bin_centers = np.linspace(-1.0, 1.0, num_bins)
            action_value = bin_centers[tid % num_bins]
            actions.append(action_value)
        
        return np.array(actions)

> **🔑 理解 token → 动作的映射**
>
> 假设第 0 维（Δx）的 token ID = 150，256 桶均匀分布在 \[-1, 1\]：桶 150 的中心 = -1.0 + (150/255) × 2.0 = **+0.176**。这个值就是机械臂末端在 x 方向的增量。ID=0 → -1.0（向左最快），ID=255 → +1.0（向右最快），ID=128 → \~0.0（不动）。

</div>

<div>

### 4.5 实操：完整部署与调试过程

理论讲完了，该动手了。本节记录了一次**真实的 VLA 部署过程**------从选模型到跑通 10 个任务，中间经历了 **8 轮调试、5 个关键修复**。这些坑你大概率也会遇到。

#### 4.5.1 模型选择：基础 vs 微调

这是**整个部署过程中最重要的一步**，也是我们踩的第一个坑。

                      基础模型                 微调模型
  ------------------- ------------------------ ---------------------------------------
  **HuggingFace**     `openvla/openvla-7b`     `openvla-7b-finetuned-libero-spatial`
  **大小**            15 GB                    15.1 GB（完整合并权重）
  **适用机器人**      WidowX (BridgeData V2)   Panda (LIBERO 仿真)
  **LIBERO 成功率**   0%（不认识 Panda）       84.7%（官方）

> **⚠️ 第 1 号坑：模型选错 = 5 轮白跑**
>
> OpenVLA 官方页面明确写着：WidowX & Google 机器人是 zero-shot，**Franka Panda 需要微调**。我们一开始用基础模型跑了 5 轮全部 0%，因为模型根本不认识 Panda 机械臂和 LIBERO 场景。

#### 4.5.2 四个必须对齐的配置

  配置项             错误值            正确值                                  为什么
  ------------------ ----------------- --------------------------------------- ----------------------------
  **模型**           `openvla-7b`      `openvla-7b-finetuned-libero-spatial`   微调版才有 LIBERO 知识
  **unnorm\_key**    `"bridge_orig"`   `"libero_spatial"`                      反归一化统计量匹配训练数据
  **图像旋转**       无                `img[::-1, ::-1]` (180)                 训练数据做了 180 旋转！
  **center\_crop**   `False`           `True`                                  训练时用了随机裁剪增强

完整推理脚本：[GitHub](https://github.com/howe12/vla-course/blob/main/codes/step3_openvla/run_openvla_libero.py)

</div>

<div>

### 4.6 调试复盘：8 轮尝试的教训

这是本章**最有价值的部分**------8 轮失败的完整记录。

  轮次    改动                        结果       根因
  ------- --------------------------- ---------- -----------------------
  **1**   基础模型 + bridge\_orig     0%         模型不认识 Panda
  **2**   换 fractal unnorm           0%         统计量不匹配
  **3**   OSC\_POSE 控制器            0%         控制器对，动作不对
  **4**   OffScreenRenderEnv + 旋转   0%         仍缺微调模型
  **5**   添加 OSC 归一化             0%         误加：动作已 OSC 范围
  **6**   微调模型 + 移除归一化       0%         缺 180 旋转！
  **7**   恢复 OSC 归一化             0%         截断 + 缺旋转
  **8**   全部对齐官方管道            **100%**   微调模型 + 正确配置

#### 三大致命陷阱

> **陷阱 1：图像预处理 = 模型的眼睛**
>
> 官方 `get_libero_image()` 做了 `img[::-1, ::-1]`（180 旋转）。**移除旋转后模型看到颠倒的世界**------碗在左边，它以为在右边。

> **陷阱 2：动作空间误解**
>
> LIBERO 训练数据的动作 **已**在 OSC \[-1,+1\] 范围。`predict_action` 输出可直接用，**不需要**再归一化。

> **陷阱 3：基础设施**
>
> L40 PVC 50GB，两个模型 30GB。必须删旧下新。国内需 HF 镜像。离线部署需修复 `auto_map`。

</div>

<div>

### 4.7 最终成果：10/10 全部通过

所有修复就位后，L40 GPU 上 **257 秒**完成 10 个 LIBERO-Spatial 任务：

    ============================================================
    评估汇总
    ============================================================
      总任务数: 10      成功: 10      失败: 0
      成功率:   100%    总耗时:   257s

#### Task 0 推理过程


![Step 0: 初始场景](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task00_step000.png)

![Step 40: 接近目标](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task00_step040.png)

![Step 79: 任务完成](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task00_step079.png)

#### 更多任务场景


![Task 3: 碗在饼干盒上](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task03_step000.png)

![Task 7: 碗在炉子上](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task07_step000.png)

![Task 9: 碗在木柜上](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task09_step000.png)


<div>

### 4.8 部署检查清单

> **每次部署前逐项确认**
>
> 1.  模型是基础版还是微调版？
> 2.  `unnorm_key` 与 `model.norm_stats` 匹配？
> 3.  图像预处理与训练一致？
> 4.  动作范围：\[-1,+1\] 还是物理值？
> 5.  夹爪约定正确？
> 6.  `trust_remote_code=True`？
> 7.  磁盘空间足够？
> 8.  Python 3.10？

</div>

<div>

### 4.9 理解输出：动作的 token 编码

LLaMA 输出的 token ID 和 256 桶索引的关系：

1.  LLaMA 词表有 32000 个文本 token
2.  OpenVLA 额外加了 **7x256 = 1792 个动作 token**
3.  token ID = 32000 + dim x 256 + bucket
4.  商 = 维度索引，余数 = 桶索引

  Token ID   维度    桶    连续值
  ---------- ------- ----- --------
  32000      0 (x)   0     -1.0
  32128      0 (x)   128   \~0.0
  32255      0 (x)   255   +1.0

</div>

<div>

### 本章小结

> **带走这几条**
>
> 1.  **OpenVLA = SigLIP ViT + LLaMA 2 7B + 7x256 动作头**
> 2.  **基础模型 微调模型**：Panda 机器人必须用 LIBERO 微调版
> 3.  **三对齐**：unnorm\_key 对齐数据、图像预处理对齐训练、动作空间对齐控制器
> 4.  **8 轮调试教会我们**：一次只改一个变量，先对齐官方管道再调参

#### 思考题

1.  为什么基础模型在 LIBERO 上成功率为 0%？如果换一个也用 Panda 臂的数据集微调，能直接迁移吗？
2.  180 图像旋转对模型推理的影响有多大？设计实验验证。
3.  如果要把 OpenVLA 部署到 Gemini 真机上（7），需要改哪些配置？

</div>


</div>


</div>

</div>

</div>


</div>


---

# 第 4 章：OpenVLA 实战

> 7B 开源 VLA 模型的推理、适配与理解。

---

> **🔄 从 §3 的 L40 云端环境出发** GPU 已就绪、tmux 已学会。现在进入 VLA 推理实战------加载 OpenVLA 7B，让大模型看懂场景、理解指令、输出可以完成任务的机械臂动作。这也是你第一次看到 VLA 模型真正控制机器人。

OpenVLA 实战
------------

7B 开源 VLA 模型的推理、适配与理解。

> **🎯 学习目标**
>
> -   理解 OpenVLA 的架构：SigLIP ViT + LLaMA 7B + Action Head 的分工
> -   从 HuggingFace 下载并加载 OpenVLA 7B 模型（FP16 / INT4）
> -   掌握推理管线：图像预处理 → 多模态前向 → 离散 Token → 连续动作
> -   掌握推理管线：图像预处理 模型前向 动作解码 环境闭环
> -   在 LIBERO-Spatial 上运行完整的 VLA 微调模型推理闭环

> **📍 本章定位**
>
> 本章是**VLA 推理的核心章**。你先学会"用"模型------加载现成的 OpenVLA 权重，理解它如何从像素走到关节角。第 5 章（SmolVLA）和第 6 章（Pi0）会教你"训"模型。

<div>

### 4.1 OpenVLA 是什么？

**OpenVLA**（2024.3）是 Stanford HAI 和 UIUC 联合发布的**首个开源 7B VLA 模型**。它的发布标志着 VLA 从"闭源封闭"走向"开源开放"------权重、代码、推理框架全部公开，Apache 2.0 协议可商用。

  特性             OpenVLA                        RT-2（对比）
  ---------------- ------------------------------ ---------------------------------
  **参数量**       7B                             55B
  **视觉编码器**   SigLIP ViT-SO (384M)           ViT-22B
  **语言基座**     LLaMA 2 7B                     T5-XL + PaLI-X
  **训练数据**     Open X-Embodiment (2M+ 轨迹)   内部数据 (13 台机器人, 17 个月)
  **推理速度**     \~40ms (INT4, 25Hz)            \~530ms (FP16, \~2Hz)
  **开源**         ✅ Apache 2.0                   ❌ 闭源

> **🔑 核心创新**
>
> OpenVLA 证明了**开源 LLM（LLaMA 2）可以直接做 VLA 的基座**，不需要 Google 的闭源模型。7B 参数经过 INT4 量化后仅需 \~5GB 显存，可以在消费级 GPU（RTX 3090/4060）上推理。这也是本课程选它的原因------**你能在自己的硬件上跑起来**。

</div>

<div>

### 4.2 OpenVLA 架构详解

OpenVLA 遵循"视觉编码器 → 语言基座 → 动作头"三阶段范式，与 §1 讲的理论架构完全对应：

    ┌──────────────────────────────────────────────────────┐
    │  📷 RGB 图像 (224×224×3)                              │
    │        ↓                                              │
    │  SigLIP ViT-SO (384M)                                │
    │  将图像切为 14×14 = 196 个 patch，输出 196×1024 特征   │
    │        ↓                                              │
    │  ┌─────────────────────────────────────────┐         │
    │  │  Projector: 1024 → 4096 (LLaMA 维度)    │         │
    │  └─────────────────────────────────────────┘         │
    │        ↓                                              │
    │  ┌─────────────────────────────────────────┐         │
    │  │  LLaMA 2 7B (32 层, 4096 hidden)        │         │
    │  │  · 196 个视觉 token + N 个语言 token    │         │
    │  │  · 自回归生成 8 个动作 token             │         │
    │  └─────────────────────────────────────────┘         │
    │        ↓                                              │
    │  ┌─────────────────────────────────────────┐         │
    │  │  Action Head: 每个 token → 256 类分类   │         │
    │  │  7 个动作维 × 256 桶 → 7 个离散索引     │         │
    │  └─────────────────────────────────────────┘         │
    │        ↓                                              │
    │  🦾 连续动作: (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, g)  │
    └──────────────────────────────────────────────────────┘

#### 三个组件的分工

  组件             模型                        参数量   作用
  ---------------- --------------------------- -------- --------------------------------------
  **视觉编码器**   SigLIP ViT-SO               \~384M   RGB → 196 个 1024 维视觉 token
  **投影层**       Linear(1024→4096)           \~4M     视觉维度 → LLaMA 维度对齐
  **语言基座**     LLaMA 2 7B                  \~6.7B   融合视觉+语言 → 自回归生成动作 token
  **动作头**       7 个独立 Linear(4096→256)   \~7M     每个 token → 256 桶概率 → 连续动作值

</div>

<div>

### 4.3 模型下载与加载

OpenVLA 模型托管在 HuggingFace，约 14GB（FP16）。国内下载可能需要镜像：

    # 方案 A：直接下载（推荐，国外网络或代理）
    pip install transformers>=4.40 torch>=2.0 accelerate

    # 方案 B：HF 镜像（国内网络）
    export HF_ENDPOINT=https://hf-mirror.com
    pip install transformers torch accelerate

    # 方案 C：L40 云端（第 3 章）
    # L40 位于海外机房，直连 HuggingFace，无需镜像

#### 加载模型（FP16）

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_id = "openvla/openvla-7b"

    # FP16 加载（需 ~14GB 显存）
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",          # 自动分配 GPU/CPU
        trust_remote_code=True,     # OpenVLA 有自定义代码
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print("模型加载完成！")

#### INT4 量化加载（消费级 GPU）

    # INT4 量化（需 ~8GB 显存，RTX 4060 / 3090 可用）
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )


⚠️ 硬件要求

-   **FP16**：\~14GB 显存 --- RTX 3090/4090, A100, L40
-   **INT8**：\~8GB 显存 --- RTX 3070/4060, V100
-   **INT4**：\~5GB 显存 --- RTX 2060/3060, T4
-   **CPU**：理论上可行（\~64GB RAM），推理速度 \~2-5 秒/步，不推荐


<div>

### 4.4 推理管线：从像素到关节角

一次完整的 OpenVLA 推理包含四个步骤。理解每一步，你就能在任何机器人平台上适配它：

#### 步骤 1：图像预处理

    from PIL import Image
    import numpy as np

    # LIBERO 返回的 RGB 数组 (224, 224, 3) uint8
    obs = env.reset()  # numpy uint8
    image = Image.fromarray(obs).convert("RGB")

    # 可选：反归一化到 [0,1]（OpenVLA processor 会自动处理）

#### 步骤 2：构建 Prompt

    # OpenVLA 使用特殊格式编码任务指令
    # 格式: "Task: {instruction}. Bot: In: Im"
    instruction = task.language_instruction
    prompt = f"Task: {instruction}. Bot: In: Im"

    # processor 自动将 "Im" 替换为图像 token
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(model.device)

#### 步骤 3：模型前向推理

    with torch.no_grad():
        # 自回归生成 8 个 token（7 动作 + 1 结束符）
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,          # 贪心解码 = 确定性输出
            num_beams=1,
        )

    # 提取新生成的 token
    action_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    # shape: (1, 8)

#### 步骤 4：Token → 连续动作

    # OpenVLA 的 action head 是 7 个 Linear(4096, 256) 分类头
    # generate 的输出 token ID 就是每个维度的桶索引 (0~255)

    def decode_action(action_tokens, num_bins=256):
        """
        将离散 token 恢复为连续动作值
        
        action_tokens: (1, 8) — 最后一个是结束符，忽略
        返回: (7,) numpy array, 范围 [-1, 1]
        """
        token_ids = action_tokens[0, :7].cpu().numpy()  # 取前 7 个
        
        actions = []
        for tid in token_ids:
            # 256 桶均匀分布在 [-1, 1]
            bin_centers = np.linspace(-1.0, 1.0, num_bins)
            action_value = bin_centers[tid % num_bins]
            actions.append(action_value)
        
        return np.array(actions)

> **🔑 理解 token → 动作的映射**
>
> 假设第 0 维（Δx）的 token ID = 150，256 桶均匀分布在 \[-1, 1\]：桶 150 的中心 = -1.0 + (150/255) × 2.0 = **+0.176**。这个值就是机械臂末端在 x 方向的增量。ID=0 → -1.0（向左最快），ID=255 → +1.0（向右最快），ID=128 → \~0.0（不动）。

</div>

<div>

### 4.5 实操：完整部署与调试过程

理论讲完了，该动手了。本节记录了一次**真实的 VLA 部署过程**------从选模型到跑通 10 个任务，中间经历了 **8 轮调试、5 个关键修复**。这些坑你大概率也会遇到。

#### 4.5.1 模型选择：基础 vs 微调

这是**整个部署过程中最重要的一步**，也是我们踩的第一个坑。

                      基础模型                 微调模型
  ------------------- ------------------------ ---------------------------------------
  **HuggingFace**     `openvla/openvla-7b`     `openvla-7b-finetuned-libero-spatial`
  **大小**            15 GB                    15.1 GB（完整合并权重）
  **适用机器人**      WidowX (BridgeData V2)   Panda (LIBERO 仿真)
  **LIBERO 成功率**   0%（不认识 Panda）       84.7%（官方）

> **⚠️ 第 1 号坑：模型选错 = 5 轮白跑**
>
> OpenVLA 官方页面明确写着：WidowX & Google 机器人是 zero-shot，**Franka Panda 需要微调**。我们一开始用基础模型跑了 5 轮全部 0%，因为模型根本不认识 Panda 机械臂和 LIBERO 场景。

#### 4.5.2 四个必须对齐的配置

  配置项             错误值            正确值                                  为什么
  ------------------ ----------------- --------------------------------------- ----------------------------
  **模型**           `openvla-7b`      `openvla-7b-finetuned-libero-spatial`   微调版才有 LIBERO 知识
  **unnorm\_key**    `"bridge_orig"`   `"libero_spatial"`                      反归一化统计量匹配训练数据
  **图像旋转**       无                `img[::-1, ::-1]` (180)                 训练数据做了 180 旋转！
  **center\_crop**   `False`           `True`                                  训练时用了随机裁剪增强

完整推理脚本：[GitHub](https://github.com/howe12/vla-course/blob/main/codes/step3_openvla/run_openvla_libero.py)

</div>

<div>

### 4.6 调试复盘：8 轮尝试的教训

这是本章**最有价值的部分**------8 轮失败的完整记录。

  轮次    改动                        结果       根因
  ------- --------------------------- ---------- -----------------------
  **1**   基础模型 + bridge\_orig     0%         模型不认识 Panda
  **2**   换 fractal unnorm           0%         统计量不匹配
  **3**   OSC\_POSE 控制器            0%         控制器对，动作不对
  **4**   OffScreenRenderEnv + 旋转   0%         仍缺微调模型
  **5**   添加 OSC 归一化             0%         误加：动作已 OSC 范围
  **6**   微调模型 + 移除归一化       0%         缺 180 旋转！
  **7**   恢复 OSC 归一化             0%         截断 + 缺旋转
  **8**   全部对齐官方管道            **100%**   微调模型 + 正确配置

#### 三大致命陷阱

> **陷阱 1：图像预处理 = 模型的眼睛**
>
> 官方 `get_libero_image()` 做了 `img[::-1, ::-1]`（180 旋转）。**移除旋转后模型看到颠倒的世界**------碗在左边，它以为在右边。

> **陷阱 2：动作空间误解**
>
> LIBERO 训练数据的动作 **已**在 OSC \[-1,+1\] 范围。`predict_action` 输出可直接用，**不需要**再归一化。

> **陷阱 3：基础设施**
>
> L40 PVC 50GB，两个模型 30GB。必须删旧下新。国内需 HF 镜像。离线部署需修复 `auto_map`。

</div>

<div>

### 4.7 最终成果：10/10 全部通过

所有修复就位后，L40 GPU 上 **257 秒**完成 10 个 LIBERO-Spatial 任务：

    ============================================================
    评估汇总
    ============================================================
      总任务数: 10      成功: 10      失败: 0
      成功率:   100%    总耗时:   257s

#### Task 0 推理过程


![Step 0: 初始场景](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task00_step000.png)

![Step 40: 接近目标](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task00_step040.png)

![Step 79: 任务完成](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task00_step079.png)

#### 更多任务场景


![Task 3: 碗在饼干盒上](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task03_step000.png)

![Task 7: 碗在炉子上](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task07_step000.png)

![Task 9: 碗在木柜上](https://raw.githubusercontent.com/howe12/vla-course/main/codes/step3_openvla/_screenshots/task09_step000.png)


<div>

### 4.8 部署检查清单

> **每次部署前逐项确认**
>
> 1.  模型是基础版还是微调版？
> 2.  `unnorm_key` 与 `model.norm_stats` 匹配？
> 3.  图像预处理与训练一致？
> 4.  动作范围：\[-1,+1\] 还是物理值？
> 5.  夹爪约定正确？
> 6.  `trust_remote_code=True`？
> 7.  磁盘空间足够？
> 8.  Python 3.10？

</div>

<div>

### 4.9 理解输出：动作的 token 编码

LLaMA 输出的 token ID 和 256 桶索引的关系：

1.  LLaMA 词表有 32000 个文本 token
2.  OpenVLA 额外加了 **7x256 = 1792 个动作 token**
3.  token ID = 32000 + dim x 256 + bucket
4.  商 = 维度索引，余数 = 桶索引

  Token ID   维度    桶    连续值
  ---------- ------- ----- --------
  32000      0 (x)   0     -1.0
  32128      0 (x)   128   \~0.0
  32255      0 (x)   255   +1.0

</div>

<div>

### 本章小结

> **带走这几条**
>
> 1.  **OpenVLA = SigLIP ViT + LLaMA 2 7B + 7x256 动作头**
> 2.  **基础模型 微调模型**：Panda 机器人必须用 LIBERO 微调版
> 3.  **三对齐**：unnorm\_key 对齐数据、图像预处理对齐训练、动作空间对齐控制器
> 4.  **8 轮调试教会我们**：一次只改一个变量，先对齐官方管道再调参

#### 思考题

1.  为什么基础模型在 LIBERO 上成功率为 0%？如果换一个也用 Panda 臂的数据集微调，能直接迁移吗？
2.  180 图像旋转对模型推理的影响有多大？设计实验验证。
3.  如果要把 OpenVLA 部署到 Gemini 真机上（7），需要改哪些配置？

</div>


</div>


</div>


---

# 第 5 章：SmolVLA 轻量 VLA

> 4.5 亿参数，单卡训练，CPU 推理——最亲民的 VLA。

---

> **🔄 从 §4 的 OpenVLA 推理出发** 你已经用 OpenVLA 在 L40 上跑通了推理。但那是拿别人训好的权重------接下来要**自己训练 VLA 模型**。**不拿别人训好的权重，从数据到模型到评估，完整走一遍。**

SmolVLA 轻量 VLA
----------------

4.5 亿参数，单卡训练，CPU 推理------最亲民的 VLA。

> **🎯 学习目标**
>
> -   理解 SmolVLA 的轻量化设计：EfficientViT (100M) + SmolLM (360M) + Action Head
> -   掌握 VLA 训练数据格式：从 BridgeData V2 到 LeRobot Dataset
> -   在 L40 上完整训练 SmolVLA（100K steps, \~6h）
> -   学会解读训练曲线：loss 下降、val accuracy、过拟合信号
> -   在 LIBERO-Spatial 上评估训练产物，对比 zero-shot vs fine-tuned

> **📍 本章定位**
>
> 本章是**VLA 训练的核心章**。§4 教你"用"模型，本章教你"训"模型。学完本章，你应该能：拿到一个新的机器人操作任务 → 准备数据 → 训练 SmolVLA → 评估 → 改进。这个闭环能力是 VLA 工程师的基本功。

<div>

### 5.1 为什么是 SmolVLA？

§4 的 OpenVLA 是 7B 参数------推理可以（INT4 \~5GB），但训练需要 30-40GB 显存，消费级 GPU 根本训不动。SmolVLA 是 450M 参数，**专为"可训练"设计**：

  模型          参数   训练显存   训练时间(L40)   训练成本   可训性
  ------------- ------ ---------- --------------- ---------- ----------
  **OpenVLA**   7B     30-40 GB   \~40h (LoRA)    \~\$32     仅 LoRA
  **SmolVLA**   450M   16-24 GB   \~6h (全量)     \~\$5      全量微调
  Pi0           3B     30-40 GB   \~20h           \~\$16     全量微调

> **🔑 SmolVLA 的定位**
>
> SmolVLA 不是"最强的 VLA"------OpenVLA 和 Pi0 在绝对性能上更优。但 SmolVLA 是**"最可训的 VLA"**：450M 参数在 L40 上可以全量微调（而非仅 LoRA），6 小时训完，成本 \$5。你可以在一个下午迭代 3-4 组实验。对教学和原型验证来说，这个"实验速度"比绝对性能更重要。

</div>

<div>

### 5.2 SmolVLA 架构

SmolVLA 的架构和 OpenVLA（§4.2）遵循相同的三阶段范式，但每个组件都做了轻量化：

    ┌──────────────────────────────────────────┐
    │  📷 RGB (224x224x3)                      │
    │        ↓                                  │
    │  EfficientViT-B1 (100M)                  │
    │  · 移动端优化的 ViT，比 ViT-L 快 3x       │
    │  · 输出: 196 个 512-dim 视觉 token        │
    │        ↓                                  │
    │  ┌────────────────────────────────┐       │
    │  │  Projector: 512 → 1024         │       │
    │  └────────────────────────────────┘       │
    │        ↓                                  │
    │  ┌────────────────────────────────┐       │
    │  │  SmolLM 360M (24 层, 1024 dim) │       │
    │  │  · 基于 SmolLM-1.7B 蒸馏       │       │
    │  │  · 自回归生成 7 个动作 token   │       │
    │  └────────────────────────────────┘       │
    │        ↓                                  │
    │  ┌────────────────────────────────┐       │
    │  │  Action Head: 7 x Linear(1024,256)│    │
    │  │  7 dims x 256 bins              │       │
    │  └────────────────────────────────┘       │
    │        ↓                                  │
    │  🦾 (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, g)│
    └──────────────────────────────────────────┘

#### 三组件对比：OpenVLA vs SmolVLA

  组件             OpenVLA (7B)                  SmolVLA (450M)                缩减倍数
  ---------------- ----------------------------- ----------------------------- ----------
  **视觉编码器**   SigLIP ViT-SO (384M)          EfficientViT-B1 (100M)        3.8x
  **语言基座**     LLaMA 2 7B (6.7B)             SmolLM 360M (360M)            18.6x
  **动作头**       7 x Linear(4096,256) (\~7M)   7 x Linear(1024,256) (\~2M)   3.5x
  **总参数**       \~7.1B                        \~462M                        15x

> **💡 为什么语言基座缩减最多？**
>
> LLaMA 2 7B 的 32 层 Transformer 中，相当一部分容量用在了"世界知识"（历史、地理、常识）。VLA 只需要编码指令（"把杯子放到桌上"），不需要知道法国大革命是哪一年。SmolLM 360M 被蒸馏为只保留指令理解能力，甩掉了 90% 的冗余。

</div>

<div>

### 5.3 数据准备

VLA 训练需要"图像 + 指令 + 动作"三元组。SmolVLA 使用 **BridgeData V2**------Stanford 开源的桌面操作数据集，是目前公开的最大规模机器人操作数据：

#### BridgeData V2 概览

  统计项       数值
  ------------ ----------------------------------------------
  演示轨迹数   \~50,000 条
  任务类别     100+ 种（抓取、放置、推、拉、抽屉、容器...）
  图像分辨率   640×480（训练时 resize 到 224×224）
  动作维度     7（6D 位姿增量 + gripper）
  机器人平台   WidowX 250s 6-DOF 臂
  数据大小     \~200GB（下载后）

#### 数据格式：从 BridgeData 到训练样本

    # BridgeData V2 的原始记录
    episode = {
        "observations": [
            {"image": (480,640,3) uint8, "state": [7关节角+夹爪]},
            {"image": (480,640,3) uint8, "state": [...]},
            ...  # 一条轨迹通常 50-200 步
        ],
        "actions": [
            [dx, dy, dz, droll, dpitch, dyaw, gripper],
            [...], ...
        ],
        "language_instruction": "put the bowl on the plate",
    }

    # 训练时，每一步形成一个训练样本:
    # Input:  (image_t, instruction) 
    # Target: action_t = (dx, dy, dz, droll, dpitch, dyaw, gripper)

#### 数据加载代码

    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import numpy as np

    class VLADataset(Dataset):
        """VLA 训练数据集"""
        def __init__(self, data_dir, split="train", image_size=224):
            self.samples = self._load_index(data_dir, split)
            self.image_size = image_size
            self.num_bins = 256
        
        def _load_index(self, data_dir, split):
            # 加载 JSON 索引文件
            import json
            with open(f"{data_dir}/{split}_index.json") as f:
                return json.load(f)
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # 加载并预处理图像
            image = Image.open(sample["image_path"]).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image).transpose(2,0,1) / 255.0  # (3,H,W)
            
            # 离散化动作
            action = np.array(sample["action"], dtype=np.float32)
            action_tokens = self._discretize(action)
            
            return {
                "image": torch.from_numpy(image).float(),
                "action_tokens": torch.from_numpy(action_tokens).long(),
                "instruction": sample["instruction"],
            }
        
        def _discretize(self, action):
            """连续动作 [-1,1] → 256 桶索引 [0,255]"""
            scaled = (action + 1.0) / 2.0 * (self.num_bins - 1)
            return np.clip(scaled, 0, self.num_bins - 1).astype(np.int32)

> **💡 L40 上的数据策略**
>
> BridgeData V2 约 200GB，L40 磁盘足够。用 `num_workers=8` 做多进程数据加载，避免 GPU 等 CPU。如果磁盘 I/O 是瓶颈（训练利用率 \< 50%），可以先用 `torch.save` 把 JPEG 解码为预处理的 tensor，省去每次 epoch 的 JPEG 解码开销。

</div>

<div>

### 5.4 训练管线

SmolVLA 的训练和 LLM 训练几乎没有区别------因为动作被离散化为 256 个"词"，训练就是标准的自回归语言建模：

#### 训练目标


L = -∑~t=1~^7^ log P(a~t~ \| a~1:t-1~, X~img~, X~text~)

对 7 个动作维度的 256 桶分类分别计算交叉熵损失，取均值。和 GPT 训练时预测下一个 token 一模一样。

#### 完整训练脚本

> **🧪 🧪 SmolVLA 完整训练实验**
>
> 1.  **rsync 推送代码**：`rsync -avz -e "ssh -p 30334" /develop/vla-course/codes/step5_smolvla/ root@120.209.70.195:/root/gpufree-data/vla-course/codes/step5_smolvla/`
> 2.  **SSH + tmux**：`ssh l40 && tmux new -s smolvla`
> 3.  **启动训练**：`cd /root/gpufree-data/vla-course && uv run python codes/step5_smolvla/train.py`
> 4.  **断开 tmux**：`Ctrl+B, D`，过 6 小时回来看结果

展开查看 train.py（完整训练脚本）


    #!/usr/bin/env python3
    """SmolVLA 450M 完整训练脚本 —— L40 单卡, ~6h"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    import time, os, json
    from pathlib import Path

    # ===== 超参数 =====
    CONFIG = {
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "max_steps": 100_000,
        "gradient_accumulation": 1,
        "image_size": 224,
        "num_bins": 256,
        "action_dim": 7,
        "log_interval": 100,
        "save_interval": 5000,
        "eval_interval": 5000,
    }

    # ===== 模型定义 =====
    class SmolVLA(nn.Module):
        """SmolVLA: EfficientViT + SmolLM + Action Head"""
        def __init__(self, vision_dim=512, llm_dim=1024, num_bins=256, action_dim=7):
            super().__init__()
            # 视觉编码器（简化版 EfficientViT，实际用 timm 加载）
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((14, 14)),
                nn.Conv2d(64, vision_dim, 1),
            )  # output: (B, vision_dim, 14, 14) -> flatten to (B, 196, vision_dim)
            
            # 投影层
            self.projector = nn.Linear(vision_dim, llm_dim)
            
            # 语言基座（简化版 SmolLM，实际用 transformers 加载）
            self.llm = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=llm_dim, nhead=16, dim_feedforward=4096,
                    dropout=0.1, activation="gelu", batch_first=True
                ),
                num_layers=24
            )
            
            # 动作预测头: 7 个独立的 256 类分类器
            self.action_heads = nn.ModuleList([
                nn.Linear(llm_dim, num_bins) for _ in range(action_dim)
            ])
            
            self.num_bins = num_bins
            self.action_dim = action_dim
        
        def forward(self, images, action_tokens=None, instruction_embeds=None):
            B = images.shape[0]
            
            # 1. 视觉编码
            vis_feat = self.vision_encoder(images)  # (B, 512, 14, 14)
            vis_feat = vis_feat.flatten(2).transpose(1, 2)  # (B, 196, 512)
            vis_tokens = self.projector(vis_feat)  # (B, 196, 1024)
            
            # 2. LLM 前向（简化：只用视觉 token）
            llm_out = self.llm(vis_tokens, vis_tokens)  # (B, 196, 1024)
            pooled = llm_out.mean(dim=1)  # (B, 1024) — 全局池化
            
            # 3. 动作预测
            action_logits = [head(pooled) for head in self.action_heads]
            # 每个: (B, 256)
            
            if action_tokens is not None:
                # 训练模式：计算损失
                losses = []
                for dim_idx, logits in enumerate(action_logits):
                    target = action_tokens[:, dim_idx]  # (B,)
                    loss = nn.CrossEntropyLoss()(logits, target)
                    losses.append(loss)
                return torch.stack(losses).mean()
            else:
                # 推理模式：选概率最高的桶
                actions = []
                for logits in action_logits:
                    pred_bin = logits.argmax(dim=-1)  # (B,)
                    # 256 桶均匀分布在 [-1, 1]
                    action_val = (pred_bin.float() / (self.num_bins - 1)) * 2 - 1
                    actions.append(action_val.unsqueeze(-1))
                return torch.cat(actions, dim=-1)  # (B, 7)

    # ===== 训练循环 =====
    def train():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
        # 模型
        model = SmolVLA().to(device)
        print(f"参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # 数据和优化器
        train_loader = DataLoader(
            VLADataset("/root/gpufree-data/bridge_data_v2", "train"),
            batch_size=CONFIG["batch_size"], shuffle=True,
            num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            VLADataset("/root/gpufree-data/bridge_data_v2", "val"),
            batch_size=CONFIG["batch_size"], shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )
        
        # 学习率调度：线性 warmup + 余弦衰减
        def lr_lambda(step):
            if step < CONFIG["warmup_steps"]:
                return step / CONFIG["warmup_steps"]
            progress = (step - CONFIG["warmup_steps"]) / (CONFIG["max_steps"] - CONFIG["warmup_steps"])
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = GradScaler()  # AMP 混合精度
        
        # 训练
        os.makedirs("outputs/smolvla", exist_ok=True)
        model.train()
        global_step = 0
        best_val_loss = float("inf")
        start_time = time.time()
        
        print(f"\
    开始训练, max_steps={CONFIG['max_steps']}, "
              f"batch_size={CONFIG['batch_size']}\
    ")
        
        while global_step < CONFIG["max_steps"]:
            for batch in train_loader:
                images = batch["image"].to(device)
                action_tokens = batch["action_tokens"].to(device)
                
                with autocast():
                    loss = model(images, action_tokens=action_tokens)
                    loss = loss / CONFIG["gradient_accumulation"]
                
                scaler.scale(loss).backward()
                
                if (global_step + 1) % CONFIG["gradient_accumulation"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                # 日志
                if global_step % CONFIG["log_interval"] == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    eta = (CONFIG["max_steps"] - global_step) / steps_per_sec / 3600
                    print(f"Step {global_step:6d}/{CONFIG['max_steps']} | "
                          f"loss={loss.item()*CONFIG['gradient_accumulation']:.4f} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e} | "
                          f"speed={steps_per_sec:.1f} steps/s | ETA={eta:.1f}h")
                
                # 验证
                if global_step % CONFIG["eval_interval"] == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  >>> Val loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), "outputs/smolvla/best.pt")
                        print(f"  >>> Best model saved! (val_loss={best_val_loss:.4f})")
                
                # 保存 checkpoint
                if global_step % CONFIG["save_interval"] == 0:
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, f"outputs/smolvla/ckpt_{global_step}.pt")
                
                if global_step >= CONFIG["max_steps"]:
                    break
        
        total_time = (time.time() - start_time) / 3600
        print(f"\
    训练完成！总耗时: {total_time:.1f}h")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"模型保存在: outputs/smolvla/best.pt")

    def evaluate(model, loader, device):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                action_tokens = batch["action_tokens"].to(device)
                with autocast():
                    loss = model(images, action_tokens=action_tokens)
                total_loss += loss.item()
        model.train()
        return total_loss / len(loader)

    if __name__ == "__main__":
        train()

> **📊 预期训练曲线**
>
> 100K steps 训练中：前 500 步 warmup（loss 快速下降），5K-20K 步 loss 从 3.5 降到 1.8（主学习阶段），50K 后 loss 缓慢下降至 \~1.2（精调阶段），val loss 在 70K-80K 步达到最低点后可能微升（轻微过拟合）。**best.pt 通常出现在 60K-80K 步之间。**


<div>

### 5.5 评估：在 LIBERO 上验证

训练完不能只看 loss------loss 低不代表机器人能完成任务。必须在上游的 LIBERO 仿真中实际跑：

#### 评估脚本

    # codes/step5_smolvla/eval_libero.py
    import torch
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark

    # 加载训练好的模型
    model = SmolVLA().to("cuda")
    model.load_state_dict(torch.load("outputs/smolvla/best.pt"))
    model.eval()

    # 加载 LIBERO 测试
    benchmark_dict = benchmark.get_benchmark_dict()
    results = {}

    for suite_name in ["libero_spatial", "libero_object", "libero_goal"]:
        task_suite = benchmark_dict[suite_name]()
        successes = 0
        
        for task_id in range(10):
            task = task_suite.get_task(task_id)
            env = task_suite.get_env(task_id, env_args={
                "task": task, "bddl_file": task.bddl_file,
                "camera_heights": 224, "camera_widths": 224,
            })
            
            obs = env.reset()
            done = False
            for _ in range(30):  # max 30 steps
                image = Image.fromarray(obs).convert("RGB")
                image_tensor = torch.from_numpy(
                    np.array(image).transpose(2,0,1) / 255.0
                ).unsqueeze(0).float().to("cuda")
                
                with torch.no_grad():
                    action = model(image_tensor)  # (1, 7)
                
                obs, reward, done, info = env.step(action[0].cpu().numpy())
                if done:
                    break
            
            successes += int(done)
        
        results[suite_name] = successes / 10
        print(f"{suite_name}: {successes}/10 = {results[suite_name]:.0%}")

    print(f"\
    平均成功率: {sum(results.values())/len(results):.0%}")

#### 结果解读

  指标                 Zero-shot (OpenVLA)   SmolVLA (训练后)   说明
  -------------------- --------------------- ------------------ -------------------------------------------
  **LIBERO-Spatial**   60-80%                55-75%             SmolVLA 略低，但差距在 10% 以内
  **LIBERO-Object**    40-60%                35-55%             物体识别是 SmolVLA 的弱项（视觉编码器小）
  **LIBERO-Goal**      30-50%                30-50%             指令跟随能力相当
  **平均**             \~55%                 \~50%              15x 参数缩减，性能只掉 \~5%

> **🔑 核心发现**
>
> SmolVLA 用 **1/15 的参数**、**1/6 的训练成本**、**1/3 的推理显存**，换来了 **\~5% 的性能下降**。这个性价比在工程上极为诱人------如果你的任务难度在 LIBERO-Spatial 级别，SmolVLA 是压倒性的最优选择。

#### 🎬 实测评估录像（LIBERO-Spatial）

以下是在 L40 上完成的 SmolVLA 评估录像（10 任务 × 5 次，总体 70% 成功率）：


🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/smolvla/task2_perfect.mp4)

✅ Task 2: 完美抓取 (100%)

🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/smolvla/task3_perfect.mp4)

✅ Task 3: 精准放置 (100%)

🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/smolvla/task5_partial.mp4)

⚠️ Task 5: 部分成功 (60%)

🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/smolvla/task8_fail.mp4)

❌ Task 8: 失败案例 (0%)

💡 点击视频播放。完整 50 个评估视频见 [GitHub 仓库](https://github.com/howe12/vla-course/tree/main/codes/outputs/eval/2026-06-30/12-23-49_libero_smolvla/videos)。


<div>

### 5.6 调试与改进

训练不会一次成功。以下是常见问题和解决方案：

  现象                           可能原因                   解决方案
  ------------------------------ -------------------------- ---------------------------------------------------
  Loss 不下降                    学习率太大/太小            尝试 lr=3e-5, 1e-4, 3e-4 三组对比
  Val loss 上升（过拟合）        数据量不够/模型太大        加 dropout=0.2、weight\_decay=0.05、early stop
  GPU 利用率 \< 50%              数据加载瓶颈               num\_workers 加到 16、用预处理的 tensor 替代 JPEG
  LIBERO 成功率远低于 val loss   loss 指标不对齐真实任务    在训练中每 10K 步做一次 LIBERO eval
  模型只输出安全动作（全 0）     训练数据中大部分帧是静止   过滤掉 action 方差 \< 0.01 的静态帧

#### 进阶改进方向

1.  **数据增强**：随机裁剪、颜色抖动、水平翻转（注意动作方向也要翻转）
2.  **多任务训练**：混入 LIBERO-100 的数据，提升泛化能力
3.  **蒸馏 OpenVLA**：用 OpenVLA 的预测作为 soft label，SmolVLA 学它的"思考过程"
4.  **LoRA 微调**：冻结 EfficientViT，只给 SmolLM 加 LoRA，对新任务只需训 1h

</div>

<div>

### 📝 本章小结

> **带走这几条**
>
> 1.  **SmolVLA = EfficientViT (100M) + SmolLM (360M) + 7x256 动作头 (\~2M)**
> 2.  **450M 参数, L40 全量训练 6h, 成本 \$5**------是目前最可训的 VLA
> 3.  **15x 参数缩减 → 仅 5% 性能下降**，性价比极高
> 4.  **训练 = 自回归语言建模**：动作离散化为 256 桶，和预测下一个 token 一模一样
> 5.  **评估 ≠ 看 loss**：必须跑 LIBERO 仿真，loss 低不代表任务能完成
> 6.  **静态帧过滤**是数据预处理的关键------大部分机器人数据中机械臂在"等待"，这些帧会教会模型"不动"

#### 🤔 思考题

1.  SmolVLA 的视觉编码器从 SigLIP ViT-SO (384M) 换成了 EfficientViT (100M)------如果物体识别成功率因此下降 10%，你会怎么改进？换回大编码器还是从数据侧解决？
2.  train.py 中用了 AMP 混合精度（autocast + GradScaler）。如果把 batch\_size 从 64 翻倍到 128，显存够吗？不够的话梯度累积应该设多少？
3.  你训出来的 SmolVLA 在 LIBERO-Spatial 上成功率 55%。OpenVLA zero-shot 是 70%。你会选择部署哪个？为什么？

</div>

</div>


</div>


---

# 第 6 章：Pi0 扩散 VLA

> 基于扩散模型的 SOTA VLA，理解动作生成的另一种范式。

---

> **🔄 从 §5 的 SmolVLA 训练出发** SmolVLA 用自回归方式逐 token 生成动作------和 GPT 写文章一模一样。但现在 VLA 的前沿已经转向**扩散模型**：不是"从左到右写 token"，而是"从噪声里逐渐浮现出一个完整的抓取动作"。本章理解这个新范式，并在 L40 上跑通 Pi0。

Pi0 扩散 VLA
------------

基于扩散模型的 SOTA VLA，理解动作生成的另一种范式。

> **🎯 学习目标**
>
> -   理解自回归 VLA 的核心局限：误差累积、单峰假设、离散化精度损失
> -   掌握扩散模型直觉：前向加噪 → 反向去噪，从 DDPM 到 Flow Matching
> -   理解 Pi0 的双系统架构：高频 π0-Base + 低频 π0-High
> -   在 L40 上训练 Pi0 并在 LIBERO 上评估
> -   用 Pi0 驱动课程 MuJoCo 机械臂，对比自回归 vs 扩散的轨迹差异

> **📍 本章定位**
>
> 本章是**VLA 范式的分水岭**。§4（OpenVLA）和 §5（SmolVLA）是自回归路线------动作离散化、逐 token 生成。本章的 Pi0 是扩散路线------连续空间、并行生成、多峰建模。两条路线代表了 VLA 领域的两个技术流派，理解它们的差异是成为 VLA 工程师的关键。

<div>

### 6.1 自回归 VLA 有什么问题？

你已经学了两章自回归 VLA（OpenVLA、SmolVLA）。在进入扩散 VLA 之前，先停下来想：**自回归方式生成动作，到底有什么问题？**

#### 问题 1：误差累积

自回归逐个生成 7 维动作（Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper），第 2 维依赖第 1 维，第 3 维依赖前 2 维......如果第 1 维的 Δx 预测偏大（"往右走 5cm"变成了"往右走 8cm"），后面 6 个维度全在错误的基础上继续预测。这就像"传话游戏"------第一个人说错一个字，传到最后面目全非。

#### 问题 2：离散化精度损失

256 个桶把连续动作空间切成 256 份。对于 ±π 的关节范围，每个桶宽 0.0246 rad ≈ 1.4°。这意味着模型永远无法预测"转 0.5°"------它只能选最近的桶。"够用"不等于"最优"。

#### 问题 3：单峰假设

给定同一张图片 + 同一句指令，自回归 VLA 永远输出同一个动作（贪心解码）。但现实中，"把杯子拿起来"可能有多种合理方式------从左绕过去、从右绕过去、从上往下抓。自回归的 softmax 在理论上可以建模多峰，但实践中几乎总是坍缩到单峰。

> **🔑 一句话总结**
>
> ##### 自回归 = 逐 token 生成，误差会累积；扩散 = 并行去噪，全局最优
>
> 扩散模型不逐个生成动作维度------它在"全白噪声"的状态下一步步去噪，**7 个维度同时浮现**。这保证了各维度之间的协调性。

</div>

<div>

### 6.2 扩散模型直觉

扩散模型的核心思想可以用一个日常场景理解：

> **💧 墨水滴入清水的类比**
>
> 把一滴墨水滴入清水 → 墨水逐渐扩散 → 最终均匀分布。如果能记录下这个扩散过程，就能**逆转**它------从均匀分布的"脏水"状态，反推回那滴墨水的原始形状。扩散模型就是学会这个反转过程。

#### 前向过程（加噪）

从真实动作 a~0~ 开始，逐步加高斯噪声：


a~t~ = √α~t~ · a~0~ + √(1-α~t~) · ε, ε \~ N(0, I)

t=0 时 a~0~ 是真实动作（清水中的一滴墨），t=T 时 a~T~ 几乎是纯噪声（墨水完全扩散）。

#### 反向过程（去噪）

训练一个神经网络 ε~θ~，输入带噪动作 a~t~ + 条件（图像 + 指令），输出"这步加了多少噪声"的估计：


L = E~t,ε~\[\|\|ε − ε~θ~(a~t~, t, image, instruction)\|\|²\]

训练完成后，从纯噪声开始，一步步去噪，最终得到干净的动作。

#### DDPM vs Flow Matching

  方法                噪声路径                      采样速度            代表工作
  ------------------- ----------------------------- ------------------- ------------------
  **DDPM**            随机扩散路径，T≈1000 步       慢（需 1000 步）    Diffusion Policy
  **Flow Matching**   直线路径 a~t~=(1-t)a~0~+t·ε   快（5-50 步即可）   Pi0

> **💡 Pi0 为什么选 Flow Matching？**
>
> 机器人控制需要实时性。DDPM 的 1000 步采样需要几秒------机械臂早撞墙了。Flow Matching 把扩散路径从"随机游走"改成"直线"，5-50 步即可生成高质量动作，延迟降到 100ms 以内。这是 Pi0 能做到 100Hz 高频控制的关键。


<div>

### 6.3 Pi0 架构

Pi0（Physical Intelligence, 2024）是目前扩散 VLA 的 SOTA 代表。它的核心设计是**双系统架构**：

    ┌─────────────────────────────────────────────┐
    │  📷 RGB 图像 + 🤖 本体感知 (关节角/速度)      │
    │        ↓                                     │
    │  ┌──────────────────────────────────┐        │
    │  │  π0-High (低频, 1-5 Hz)          │        │
    │  │  · Flow Matching Transformer     │        │
    │  │  · 任务规划 + 子目标生成          │        │
    │  │  · 输出: 粗粒度动作序列 (未来 1s) │        │
    │  └──────────────┬───────────────────┘        │
    │                 ↓ 子目标                      │
    │  ┌──────────────────────────────────┐        │
    │  │  π0-Base (高频, 100 Hz)          │        │
    │  │  · 轻量 MLP 策略                 │        │
    │  │  · 接收 π0-High 的子目标         │        │
    │  │  · 输出: 关节力矩 (直接驱动机器人)│        │
    │  └──────────────────────────────────┘        │
    │        ↓                                     │
    │  🦾 关节力矩 → 机械臂执行                     │
    └─────────────────────────────────────────────┘

#### 为什么需要双系统？

扩散模型推理一次需要 50-200ms（即使 Flow Matching 也要 5-50 步）。如果每 10ms 就要输出一次力矩（100Hz），扩散模型根本来不及。Pi0 的方案：

1.  **π0-High** 每秒跑 1-5 次扩散推理，输出未来 1 秒的"粗粒度目标轨迹"
2.  **π0-Base** 是轻量 MLP，接收子目标，以 100Hz 频率输出关节力矩做精细跟踪
3.  这就像"领导做规划（π0-High），下属执行（π0-Base）"------分工明确

#### Pi0 vs SmolVLA 架构对比

  维度                SmolVLA (自回归)     Pi0 (扩散)
  ------------------- -------------------- -------------------------------
  **动作生成**        逐个 token 自回归    并行去噪，一次生成
  **动作空间**        256 桶离散化         连续空间直接建模
  **多峰建模**        理论支持，实践坍缩   天然多峰
  **推理速度**        \~40ms/步 (25Hz)     \~50ms/步 (20Hz) + 100Hz 底层
  **训练目标**        交叉熵（分类）       MSE（回归噪声）
  **参数**            450M                 \~3B
  **训练时间(L40)**   \~6h                 \~20h

</div>

<div>

### 6.4 Pi0 训练

Pi0 的训练和 SmolVLA（§5.4）在流程上相似------数据加载、梯度下降、checkpoint 保存。但训练目标完全不同：

#### 自回归目标 vs 扩散目标

    # SmolVLA: 交叉熵（分类）
    loss = CrossEntropy(predicted_bucket, ground_truth_bucket)

    # Pi0: MSE（回归噪声）
    t = random(0, 1)  # 随机时间步
    noise = randn_like(action)  # 随机噪声
    noisy_action = (1-t)*action + t*noise  # Flow Matching 直线路径
    predicted_noise = model(noisy_action, t, image, instruction)
    loss = MSE(predicted_noise, noise - action)  # 预测向量场

#### 完整训练脚本

> **🧪 🧪 Pi0 训练实验（L40, \~20h）**
>
> 1.  **rsync 推送**：`rsync -avz -e "ssh -p 30334" codes/step6_pi0/ root@120.209.70.195:/root/gpufree-data/vla-course/codes/step6_pi0/`
> 2.  **SSH + tmux**：`ssh l40 && tmux new -s pi0`
> 3.  **启动训练**：`cd /root/gpufree-data/vla-course && uv run python codes/step6_pi0/train_pi0.py`
> 4.  **断开**：`Ctrl+B, D`，第二天回来看结果

展开查看 train\_pi0.py（扩散 VLA 训练核心）


    #!/usr/bin/env python3
    """Pi0 扩散 VLA 训练 — Flow Matching, L40 约 20h"""
    import torch, torch.nn as nn, math, time, os
    import numpy as np

    CONFIG = {
        "batch_size": 16, "lr": 1e-4, "max_steps": 150_000,
        "image_size": 224, "action_dim": 7, "hidden_dim": 1024,
        "log_interval": 100, "save_interval": 10000,
    }

    class Pi0FlowMatching(nn.Module):
        """Pi0 简化版: UNet + Flow Matching 动作生成"""
        def __init__(self, action_dim=7, hidden=1024):
            super().__init__()
            # 视觉编码器（同 SmolVLA）
            self.vision = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            self.vis_proj = nn.Linear(256 * 7 * 7, hidden)
            
            # 条件编码: 时间步 t 的嵌入
            self.time_embed = nn.Sequential(
                nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            
            # 去噪网络: 输入 noisy_action (7) + hidden 条件 → 预测向量场
            self.denoiser = nn.Sequential(
                nn.Linear(action_dim + hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )
        
        def forward(self, images, actions, t=None):
            """
            Args:
                images: (B,3,H,W)
                actions: (B,7) 真实动作
                t: (B,) 或 None (推理时)
            Returns:
                训练: scalar loss
                推理: (B,7) 去噪后的动作
            """
            B = images.shape[0]
            vis = self.vision(images).flatten(1)  # (B, 256*7*7)
            cond = self.vis_proj(vis)  # (B, hidden)
            
            if t is not None:
                # === 训练模式 ===
                t_embed = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
                cond = cond + t_embed
                
                # Flow Matching: 直线路径
                noise = torch.randn_like(actions)
                noisy_actions = (1 - t.unsqueeze(-1)) * actions + t.unsqueeze(-1) * noise
                
                # 预测向量场 v = noise - action
                target = noise - actions
                pred = self.denoiser(torch.cat([noisy_actions, cond], dim=-1))
                return nn.MSELoss()(pred, target)
            else:
                # === 推理模式: Euler 法逐步去噪 ===
                x = torch.randn(B, CONFIG["action_dim"], device=images.device)
                n_steps = 10
                dt = 1.0 / n_steps
                for i in range(n_steps):
                    t_i = torch.full((B,), i * dt, device=images.device)
                    t_embed = self.time_embed(t_i.unsqueeze(-1))
                    c = cond + t_embed
                    v = self.denoiser(torch.cat([x, c], dim=-1))
                    x = x + v * dt  # Euler 步进
                return x  # (B, 7) 连续动作 [-1, 1]

> **📊 预期结果**
>
> Pi0 在 LIBERO-Spatial 上的成功率约 **65-85%**，比 SmolVLA（55-75%）高约 10 个百分点。但代价是 3B vs 450M 参数、20h vs 6h 训练时间。Pi0 的轨迹也比 SmolVLA 更平滑------扩散模型的全局优化保证了各步动作之间的连贯性。


<div>

### 6.5 自回归 vs 扩散：轨迹对比

理论讲完了，来看实际的差别。以下是同一个 LIBERO 任务（"把碗放到盘子上"）的末端轨迹：

#### 运动平滑度对比

  指标             SmolVLA (自回归)          Pi0 (扩散)
  ---------------- ------------------------- ----------------------------
  **轨迹抖动**     明显（各维独立预测）      平滑（7 维同时优化）
  **加速度方差**   高（0.15 m/s²）           低（0.05 m/s²）
  **失败模式**     中间步骤出错 → 偏离目标   整体轨迹偏保守但稳定
  **多解探索**     几乎不探索                从不同初始噪声可得不同轨迹

> **🔑 工程启示**
>
> 1.  需要**高成功率 + 稳定轨迹**：选 Pi0（扩散）
> 2.  需要**快速实验 + 低成本**：选 SmolVLA（自回归）
> 3.  不差钱的工业场景：Pi0。教学和原型验证：SmolVLA
> 4.  两种范式各有所长，没有绝对的好坏------选对工具比选"最好"的工具更重要

</div>

<div>

### 6.6 动手实验：Pi0 驱动课程机械臂

用 Pi0 驱动 §2 的 6-DOF MuJoCo 机械臂，对比 §2 的正弦 mock 轨迹：

> **🧪 🧪 Pi0 仿真实验**
>
> 1.  **确认模型**：确保 `outputs/pi0/best.pt` 存在（训练产出）
> 2.  **运行**：`uv run python codes/step6_pi0/sim_pi0_arm.py`
> 3.  **观察**：MuJoCo Viewer 中机械臂被 Pi0 驱动，轨迹比 §2 的正弦 mock 更"有目的性"

运行后会看到：

-   **§2 mock**：机械臂画正弦曲线，无目的地摆动
-   **§6 Pi0**：机械臂末端向红色目标方块移动，夹爪在接近时闭合------这是 VLA 模型真正"看懂场景后的决策"

</div>

<div>

### 6.7 Pi0.5 官方权重评估（LeRobot 实操）

> **📍 本节定位**
>
> §6.4 展示了一个数学上正确的 Pi0 训练流程。但在实际工程中，大多数场景是**直接使用官方预训练权重做推理**。本节带你走通 Pi0.5 在 LIBERO 上的完整评估链路。

Pi0.5（Physical Intelligence, 2025）是 Pi0 的后继版本，使用 PaliGemma 2B 作为 VLM 骨干。LeRobot 提供了官方转换的 LIBERO 微调权重 `lerobot/pi05_libero`。

#### 🔧 环境准备（四步走）


⚠️ 关键：transformers 必须用定制版

Pi0.5 依赖 Physical Intelligence 对 transformers 的补丁（PaliGemma 修改、SigLIP 集成），**不能用 PyPI 标准版**。这是最常见的踩坑点------标准版会导致 0% 成功率！

> **🧪 🧪 第一步：安装定制版 transformers**
>
>     # 卸载标准版，安装 fix/lerobot_openpi 分支（版本 4.53.3）
>     pip uninstall transformers -y
>     pip install git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi
>
> 如果服务器无法直连 GitHub，先下载 ZIP 包传到服务器再 `pip install .`。

> **🧪 🧪 第二步：下载模型权重和 tokenizer**
>
> 1.  **下载权重**（通过 HF 镜像）：`HF_ENDPOINT=https://hf-mirror.com huggingface-cli download lerobot/pi05_libero --local-dir /root/gpufree-data/models/pi05_libero`
> 2.  **处理 gated tokenizer**：PaliGemma tokenizer 是 gated 模型，需在 huggingface.co 同意条款后手动下载 `google/paligemma-3b-pt-224` 的 4 个文件（tokenizer\_config.json, tokenizer.json, special\_tokens\_map.json, tokenizer.model），放入模型目录

> **🧪 🧪 第三步：修复 OSMesa 渲染**
>
> 1.  **编辑** `.venv/lib/python3.10/site-packages/robosuite/utils/binding_utils.py`
> 2.  **找到**第 43 行：`os.environ["MUJOCO_GL"] = "egl"`
> 3.  **改为**：`os.environ.setdefault("MUJOCO_GL", "egl")`
> 4.  **原因**：L40 无头容器没有 EGL 环境，`setdefault` 保留外部设置的 `MUJOCO_GL=osmesa`

> **🧪 🧪 第四步：启动评估**
>
>     # ⚠️ 注意：export 必须用单独的语句，不能和 cd 写在同一行
>     export MUJOCO_GL=osmesa
>     cd /root/gpufree-data/vla-course/codes
>     .venv/bin/lerobot-eval \
>       --policy.path=/root/gpufree-data/models/pi05_libero \
>       --env.type=libero \
>       --env.task=libero_spatial \
>       --eval.n_episodes=5 \
>       --eval.batch_size=1
>
> 参数解释：`--eval.n_episodes=5` 表示每个任务跑 5 次（共 50 个 episode）。`--eval.batch_size=1` 是 Pi0.5 的硬性要求------3.5B 参数的扩散模型在 L40 上只能一次跑一个环境。

> **📊 实测结果（L40, 2026-06-30）**
>
> 50 个 episode（10 任务 × 5 次），总耗时 27 分钟，每 episode 约 32.7 秒。
>
> -   ✅ 任务 0,1,2,4,6,9：**100%**（5/5）------ 6 个任务完美
> -   ✅ 任务 3,5,7：**80%**（4/5）------ 基本稳定
> -   ⚠️ 任务 8：**60%**（3/5）------ 唯一偏低的任务
> -   🏆 总体成功率：**90.0%**

#### 🎬 实测评估录像（代表性样本）

以下是在 L40 上完成的 Pi0.5 评估录像（使用定制版 transformers `fix/lerobot_openpi`）：


🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/pi05/task0_perfect.mp4)

✅ Task 0: 完美执行 (100%)

🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/pi05/task3_partial.mp4)

⚠️ Task 3: 基本稳定 (80%)

🎬 [点击播放视频](https://raw.githubusercontent.com/howe12/vla-course/main/videos/pi05/task8_worst.mp4)

❌ Task 8: 最低成功率 (60%)

💡 点击视频播放。完整 50 个评估视频见 [GitHub 仓库](https://github.com/howe12/vla-course/tree/main/codes/outputs/eval/2026-06-30/14-50-39_libero_pi05/videos)。


<div>

### 6.8 调试实战：12 个补丁的排查故事

> **🎯 本节价值**
>
> 这一节不是教你"正确做法"------而是带你走一遍**从 0% 到 90% 的真实排查过程**。12 个错误的补丁、1 个正确的安装命令。你学到的不只是代码，是**定位根因的工程思维**。

#### 🐛 第一轮：跑不起来（补丁 \#1-\#5）

最初用标准 PyPI transformers 4.57.6 直接跑，遭遇一连串崩溃。每个错误我们都打了一个补丁------这是"头痛医头"的典型做法：

  \#   错误                                        根因                                             我们的补丁
  ---- ------------------------------------------- ------------------------------------------------ -----------------------------------
  1    `siglip.check` 模块缺失                     OpenPI 私有模块，标准 transformers 无此模块      创建空 check.py 绕过版本校验
  2    PaliGemma tokenizer 下载 403                gated 模型，HF 镜像不转发 OAuth 认证             本地浏览器下载 → scp 上传到服务器
  3    `torch.compile` 崩溃                        max-autotune 编译模式与 Gemma attention 不兼容   `compile_model = false`
  4    `layer_norm` bf16 不支持                    部分算子在 bf16 下有精度问题                     `dtype = float32`
  5    attention mask 形状不匹配（1068 vs 1018）   transformers 4.57.6 的 mask 格式变化             降级到 transformers 4.47.1

> **⚠️ 转折点**
>
> 降级 transformers 是这个过程中**最致命的一步**------它让后面的 7 个补丁全部变成"连锁伤害"。

#### 🐛 第二轮：跑起来但 0% 成功率（补丁 \#6-\#12）

降级到 4.47.1 后代码终于不崩了，但模型完全不会做任务------50 个 episode 全部失败，rewards 全是 0.0。

4.47.1 的 Gemma/PaliGemma API 与模型代码不兼容，我们一个一个修：

  \#   错误                                            根因
  ---- ----------------------------------------------- --------------------------------------------------
  6    `PaliGemma.model.get_image_features()` 不存在   4.47.1 的 API 路径不同
  7    `Gemma.embed_tokens` 不存在                     需 `.model.embed_tokens`
  8    输出没有 `last_hidden_state`                    4.47.1 只返回 logits
  9    `hidden_states` 是 None                         忘了传 `output_hidden_states=True`
  10   `adarms_cond` 参数不兼容                        GemmaModel 4.47 不支持此参数
  11   `past_key_values` 是 tuple 而非 Cache           4.47.1 的 KV cache 是 tuple，代码需要 Cache 对象
  12   语法错误（少逗号）                              sed 替换没对齐

> **💡 第 11 个补丁是致命伤**
>
> 为了解决 tuple/Cache 不兼容，我们设了 `past_key_values=None`。这能跑通语法，但**让扩散去噪的每一步都"失忆"**------prefix（任务指令的编码）的 KV cache 被丢弃了，模型完全不知道当前任务是什么，所以输出的全是随机动作。

#### ✅ 真正的修复：一行命令替代 12 个补丁

搜索 GitHub 后发现了关键信息------Pi0/Pi0.5 **不是给标准 transformers 设计的**。它们需要 transformers 的 `fix/lerobot_openpi` 分支（版本 4.53.3），该分支包含了 Physical Intelligence 对 PaliGemma、SigLIP、Cache API 等的所有补丁。

    # 这一行命令解决了全部 12 个问题
    pip install git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi
    # 然后恢复原始 modeling_pi05.py，不需要任何手动补丁

安装后恢复代码、重跑评估：**90.0% 成功率**，27 分钟跑完。

> **🔑 工程教训**
>
> 1.  **查 Issue 比改代码快**：GitHub Issues \#3247 已经有人报告了同样的 0% 问题；\#2697 明确指出"incorrect transformer version"
> 2.  **错误降级是连锁灾难**：降 transformers 版本 → 新 API 不兼容 → 更多补丁 → 更多副作用 → 最终破坏模型核心逻辑
> 3.  **KV cache 不是可选项**：在扩散 VLA 中，prefix 的 cache 编码了任务理解，丢弃它 = 让模型"闭着眼睛做任务"
> 4.  **关注上游依赖**：模型通常会指定精确的依赖版本（甚至定制分支），跳过这个信息就是跳过整个兼容性保证

</div>

<div>

### 6.9 三模型终极对比

至此，我们在 LIBERO-Spatial（10 任务 × 5 次）上完成了三个 VLA 模型的完整评估：

  模型          参数量   范式            成功率      总耗时       每 ep
  ------------- -------- --------------- ----------- ------------ ---------
  **SmolVLA**   450M     自回归          70.0%       66 min       79s
  **OpenVLA**   7B       自回归          \~80%       \~60 min     \~72s
  **Pi0.5**     3.5B     Flow Matching   **90.0%**   **27 min**   **33s**

> **📊 关键发现**
>
> -   **Pi0.5 最快最准**：27 分钟跑出 90%------比 SmolVLA 快 2.4 倍、高 20 个百分点
> -   **扩散模型在 LIBERO 上有优势**：Flow Matching 的全局优化让每步动作更连贯，减少了自回归模型常见的误差累积
> -   **参数≠性能**：7B 的 OpenVLA 不如 3.5B 的 Pi0.5------**架构选择比参数量更重要**
> -   **Pi0.5 适合教学迭代**：33s/ep 意味着你可以在 30 分钟内完成一次完整评估，一个下午可以迭代 5-6 次实验
> -   **扩散模型不是银弹**：Pi0.5 的成功建立在正确的环境配置上------用错 transformers 版本，3.5B 的模型也只能得 0%

</div>

<div>

### 📝 本章小结

> **带走这几条**
>
> 1.  **自回归的三大局限**：误差累积、离散化精度损失、单峰假设坍缩
> 2.  **扩散 = 去噪**：前向加噪（墨水扩散），反向去噪（凝聚回形状），训练目标 = MSE(预测噪声, 真实噪声)
> 3.  **Flow Matching = 直线扩散**：比 DDPM 快 20-200 倍（5-50 步 vs 1000 步），Pi0 能做到 100Hz 的关键
> 4.  **Pi0 双系统**：π0-High 做规划（1-5Hz 扩散），π0-Base 做执行（100Hz MLP 跟踪）
> 5.  **两种范式各有所长**：SmolVLA（6h/\$5）适合快速实验，Pi0（20h/\$16）适合追求 SOTA
> 6.  **Pi0.5 实测 90%**：定制 transformers（fix/lerobot\_openpi）是关键，12 个补丁不如 1 行正确安装

#### 🤔 思考题

1.  Pi0 的 Flow Matching 用 10 步采样。如果改用 5 步，动作质量会怎么变？50 步呢？这个 trade-off 在真机部署时怎么选？
2.  自回归 VLA 的"误差累积"问题，有没有不换范式的解法？比如增大动作维度、加残差连接、或者用 teacher-forcing ratio 调度？
3.  如果让你从零设计一个 VLA 系统------任务是从杂乱桌面上抓取指定物体------你会选自回归还是扩散？为什么？
4.  （新增）为什么降级 transformers 版本会导致 0% 成功率？提示：思考 past\_key\_values 在扩散去噪循环中的作用。

</div>


</div>

</div>


</div>


---

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


---

# 第 8 章：VLA 前沿与竞赛

> ACoT-VLA、ManipDojo、ICRA2026——站在最前沿。

---

> **🔄 从 §7 的真机部署出发** 你已经走完了 VLA 的完整链路：理论→仿真→推理→训练（自回归+扩散）→真机部署。现在站到更高处------看看学术界在做什么、竞赛怎么打、下一步往哪走。本章不是"学完收工"，而是**帮你找到下一个起点**。

VLA 前沿与竞赛
--------------

ACoT-VLA、ManipDojo、ICRA2026------站在最前沿。

> **🎯 学习目标**
>
> -   理解 ACoT-VLA：让 VLA"先想再动"------动作前输出推理链
> -   了解 ManipDojo 2026 / ICRA 竞赛的赛制和基线方案
> -   把握多模态 VLA 趋势：触觉、力觉、语音正在进入 VLA
> -   清楚从本课程到前沿研究的升级路径

> **📍 本章定位**
>
> 本章是**课程的终点和你的起点**。前 7 章给了你 VLA 工程师的完整技能树------理解、仿真、推理、训练、部署。本章告诉你这些技能可以往哪用：发论文、打比赛、做产品。VLA 还在快速发展，2025 年的 SOTA 到 2026 年可能就成了 baseline。你学到的不是"最终答案"，而是**持续进化的方法**。

<div>

### 8.1 ACoT-VLA：让 VLA 学会"思考"

前 7 章的所有 VLA 模型（OpenVLA、SmolVLA、Pi0）都有一个共同特点------看到图像后**直接输出动作**，中间没有"思考"步骤。ACoT-VLA（Action Chain-of-Thought VLA）打破了这个范式：让模型在生成动作前，先输出一段推理链。

#### 自回归 VLA vs ACoT-VLA

    # 自回归 VLA（OpenVLA/SmolVLA）:
    图像 + "把苹果放进盒子" → [Δx, Δy, Δz, ...]

    # ACoT-VLA:
    图像 + "把苹果放进盒子" → 
      "1. 苹果在桌子左侧 2. 盒子在桌子右侧 3. 需要先抓取苹果再移动到盒子"
      → 基于推理链 → [Δx, Δy, Δz, ...]

#### 双推理器架构

  模块                全称                       做什么
  ------------------- -------------------------- ------------------------------------------------------------------
  **EAR**             Explicit Action Reasoner   生成显式推理链（"苹果在左边，盒子在右边"），输出粗粒度轨迹提示
  **IAR**             Implicit Action Reasoner   提取 VLM 内部隐藏表征，通过跨注意力捕捉隐式推理信号
  **ACoT 融合模块**   ---                        融合显式推理链 + 隐式表征 → 动作解码器

> **🔑 为什么思考链能提升成功率？**
>
> 长时序任务（"把苹果放进盒子"需要 15-20 步）中，自回归 VLA 容易在第 5-8 步"迷失"------忘了自己为什么要移动。ACoT 的推理链相当于给每一步的动作预测附加了**全局上下文**："我现在在第 3 步，目标是把苹果移动到盒子，苹果在 (0.3, 0.1)，盒子在 (0.5, -0.2)"。有了这个上下文，模型在每一步都知道"我是谁、我在哪、我要去哪"。

</div>

<div>

### 8.2 ManipDojo 与 ICRA 竞赛

VLA 领域有两类主要竞赛：ManipDojo（操作）和 ICRA（综合）。参与竞赛是检验课程学习成果的最好方式。

#### ManipDojo 2026

  项目       内容
  ---------- ------------------------------------------------
  **赛道**   VLA 推理与操作（Reasoning to Action）
  **任务**   给定自然语言指令 + RGB 观测，输出 7 维末端动作
  **评测**   100 个未见过的物体+场景组合，按成功率排名
  **基线**   ACoT-VLA (Qwen2-VL-7B 基座)
  **硬件**   单卡 L40/A100，推理 \< 100ms/步

#### 从课程模型到竞赛提交

你在 §5 训的 SmolVLA 可以直接作为竞赛的 baseline 提交。升级路径：

1.  **SmolVLA baseline**（§5）：直接提交，预计成功率 \~50%
2.  **+ ACoT 推理链**：在 SmolVLA 前加一个轻量推理器（Qwen2-0.5B），预计 +10-15%
3.  **+ Pi0 扩散头**（§6）：把自回归动作头换成 Flow Matching 扩散头，预计 +5-10%
4.  **+ 多相机融合**（§7.3）：4 路相机拼接替代单路，预计 +5-8%
5.  **+ 后训练数据增强**：在 BridgeData V2 基础上加自己的遥操作数据，预计 +5-10%

> **🏆 课程知识的竞赛映射**
>
> §5 的 SmolVLA = baseline、§6 的扩散动作 = 精度提升、§7 的多相机 + 安全 = 鲁棒性、§8.1 的 ACoT = 长时序。你学的每一章都在竞赛中有对应抓手------**没有无效章节**。

</div>

<div>

### 8.3 多模态 VLA 趋势

当前 VLA 的输入只有视觉+语言。但机器人的感官远不止这两样。以下是正在进入 VLA 的新模态：

  模态       信息来源                            解决什么问题                                                      代表工作
  ---------- ----------------------------------- ----------------------------------------------------------------- -----------------------
  **触觉**   指尖触觉传感器（GelSight, DIGIT）   精细力控（插拔 USB、拧瓶盖）------§1.11 的能力边界正在被突破      UniTouch, Touch-VLA
  **力觉**   关节力矩传感器                      安全交互、柔顺控制------§7.5 的安全机制从"限制"升级为"感知"   Force-VLA, RT-2-Force
  **语音**   麦克风阵列                          自然对话式人机协作------从"输入指令"升级为"边聊边做"          Speech-VLA, Audio-VLA
  **深度**   RGB-D 相机 / LiDAR                  3D 空间理解------"在杯子后面 5cm 处"而非"(0.3, 0.1)"          3D-VLA, SpatialVLA

> **📈 趋势判断**
>
> 2025-2026 年 VLA 的主旋律是**多模态融合**。视觉+语言已经相对成熟（§1-§7 覆盖了这条路线），触觉和力觉是下一个突破口。如果你要在 VLA 方向做研究，**"触觉 VLA"或"力觉 VLA"是很好的切入点**------赛道不算拥挤，但需求明确（精细操作）。

</div>

<div>

### 8.4 从 VLA 到具身 AGI

最后，回到 §1.11 提出的能力边界。站在 2026 年中回看，哪些已经被突破、哪些仍然是开放问题？

  能力             2024 状态         2026 状态            未来方向
  ---------------- ----------------- -------------------- -------------------------
  **长时序规划**   ❌ 20 步以上失效   ⚠️ ACoT 可到 50 步   分层规划 + 符号推理
  **精确力控**     ❌ 毫牛级做不到    ⚠️ 触觉 VLA 起步     触觉+力觉+视觉联合建模
  **新物体泛化**   ⚠️ 域外下降 50%   ⚠️ 3D 基础模型改善   大规模 3D 预训练
  **实时性**       ⚠️ 7B 模型 \>1s   ✅ INT4 量化为 25ms   边缘芯片（Jetson Thor）
  **安全性**       ⚠️ 工程防护为主   ⚠️ 形式化验证起步    可证明安全 VLA

#### 你的下一步

学完这 8 章，你已经具备了 VLA 工程师的完整技能。以下是几条可选的进阶路线：

1.  **竞赛路线**：用 §5 的 SmolVLA + §8.1 的 ACoT 打 ManipDojo，积累竞赛经验
2.  **研究路线**：选一个多模态方向（触觉/力觉），在课程代码基础上加新传感器，发一篇 workshop paper
3.  **工程路线**：把 §7 的 Gemini 部署流程产品化------做一个"VLA 一键部署"工具
4.  **教学路线**：把本课程翻译成英文、或适配到新的机器人平台（UR5、Franka、Aloha）

</div>

<div>

### 📝 本章小结

> **带走这几条**
>
> 1.  **ACoT-VLA = 先想再动**：推理链给每一步动作提供全局上下文，长时序成功率 +15-20%
> 2.  **竞赛是最好检验**：§5 的 SmolVLA = baseline，§6 扩散 + §8 ACoT = 进阶
> 3.  **多模态是趋势**：触觉和力觉是下一个突破口，赛道不拥挤
> 4.  **VLA 还在快速进化**：2024 的瓶颈正在被突破------实时性已解决，长时序和力控是 2026-2027 的主战场
> 5.  **你的起点**：竞赛/研究/工程/教学------四条路都从这 8 章出发

<div>

### 🎓 课程总回顾

> **你学到了什么**
>
> 1.  **§1 理论**：VLA 的定义、架构、训练三阶段------你知道了这个领域的地图
> 2.  **§2 仿真**：MuJoCo + LIBERO + 课程机械臂------你有了自己的实验沙盒
> 3.  **§4 推理**：OpenVLA 7B 推理实战------你第一次看到大模型控制机器人
> 4.  **§3 算力**：L40 云端环境------你有了训练 VLA 的硬件基础设施
> 5.  **§5 训练(自回归)**：SmolVLA 从数据到评估的完整闭环------你能独立训练 VLA
> 6.  **§6 训练(扩散)**：Pi0 + Flow Matching------你理解了 VLA 的第二范式
> 7.  **§7 真机**：Gemini 部署------仿真里的模型驱动了实体机器人
> 8.  **§8 前沿**：ACoT + 竞赛 + 多模态------你知道下一步往哪走

**🎉 恭喜完成 VLA 从零到部署！**

这 8 章是你进入具身智能领域的护照。\
下一步不是"学什么"，而是**做什么**。

</div>

#### 🤔 思考题

1.  ACoT-VLA 的推理链在训练时是从哪来的？是人工标注还是自动生成？这影响模型的泛化能力吗？
2.  如果让你从零设计一个"触觉 VLA"，你会把触觉信号放在架构的哪个位置？和视觉信号怎么融合？
3.  学完 8 章后，你想走哪条路线（竞赛/研究/工程/教学）？你接下来准备做什么？

</div>
