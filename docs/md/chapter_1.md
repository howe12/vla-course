# 第 1 章：VLA 基础与架构

> 视觉-语言-动作模型的核心概念、技术演进与系统架构。

---

<div>

### ▎第 0 步：准备工作

在开始 VLA 课程之前，先花 5 分钟把工作环境搭好------后面 8 章的所有实验都在这个目录里跑。

> **🧪 🧪 第一次运行：创建工程目录**
>
> 1.  **创建工作目录**：`mkdir -p ~/vla-course/codes && cd ~/vla-course/codes`
> 2.  **初始化 Python 环境**：`uv venv`（uv 是新一代 Python 包管理器，比 pip 快 10-100x）
> 3.  **如果没装 uv**：`curl -LsSf https://astral.sh/uv/install.sh | sh`
> 4.  **验证**：`uv run python -c "print('✅ 环境就绪')"`

> **📁 约定的目录结构**
>
> 本课程约定工作根目录为 `~/vla-course/codes/`（你可以用自己喜欢的位置，只需把后续命令中的路径替换即可）。所有实验脚本按章节分目录：`step1_basics/`、`step2_sim/`、`step3_l40/` 等。每个实验都有独立的 Python 脚本------你只需 `uv run python stepX_xxx/实验名.py` 就能跑。

> **💡 两种运行方式任选一种**
>
> 本课程统一使用 `uv run python` 运行脚本（自动使用项目虚拟环境，无需手动激活）。如果你习惯传统方式，也可以 `source .venv/bin/activate && python`------效果一样，一个命令的事。

</div>

'

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

### 1.3 VLA 发展历程

VLA 经历了从**两阶段独立策略**到**端到端统一策略**的演进：

  阶段                时间        特征                                                             代表工作
  ------------------- ----------- ---------------------------------------------------------------- ----------------------
  **1. 两阶段**       2019-2022   视觉感知和动作控制分开处理，CLIP+BC 等                           CLIPort, ViLD
  **2. 统一**         2022-2023   RT-1 首次用 Transformer 统一三模态；RT-2 将 VLM 与动作统一训练   RT-1, RT-2
  **3. 开源规模化**   2023-2025   OpenVLA 开源 7B；π0 扩散动作；SmolVLA 极致轻量                   OpenVLA, π0, SmolVLA

  时间     模型          机构                    核心贡献
  -------- ------------- ----------------------- ---------------------------------------
  2022.7   **RT-1**      Google                  首次用 Transformer 统一视觉-语言-动作
  2023.7   **RT-2**      Google                  VLM+动作统一训练，离散化动作空间
  2024.3   **OpenVLA**   Stanford+UIUC           开源 7B VLA，HuggingFace 发布
  2024.7   **π0**        Physical Intelligence   扩散模型生成动作，专为人形机器人设计
  2025     **SmolVLA**   HuggingFace             450M 参数，消费级 GPU 可训练

> **📈 趋势**
>
> 从大而全（RT-2 55B）→ 开源可用（OpenVLA 7B）→ 极致轻量（SmolVLA 450M）。动作生成从**离散分类**（RT-1/RT-2）→ **连续回归+扩散**（π0），精度和泛化能力持续提升。

</div>

<div>

### 1.4 VLA 核心架构：三模态输入

VLA 同时接收**视觉观察**、**语言指令**和**机器人状态（可选）**三种输入：

#### （1）视觉输入

来自机器人 RGB/RGB-D 相机，经视觉编码器得到特征序列：


**V** = VisualEncoder(**X**~img~) = \[**v**~1~, **v**~2~, \..., **v**~N~\]

  编码器         特点                           代表模型
  -------------- ------------------------------ ----------
  **CLIP ViT**   大规模图文预训练，强语义理解   RT-2
  **SigLIP**     CLIP 增强版，对齐能力更强      OpenVLA
  **DINOv2**     自监督 ViT，细粒度视觉特征     SmolVLA

#### （2）语言输入

任务指令（如"拿起红色杯子"），经 LLM 编码为语言 Token。与 VLM 不同，VLA 中的语言输入主要承担**任务指令编码**的作用，而非开放式对话。

#### （3）机器人本体感知（可选）

部分 VLA 还会接收本体状态：关节角度 **q**、末端位姿 **p**、夹爪开合 g。三者拼接后送入 LLM 统一处理：


**F**~combined~ = \[**V**; **L**; **S**\]

                         ┌──────────────────────┐
    图像输入 → [视觉编码器] → 视觉 Token 序列 V ─┐
                                                │
    文本指令 → [LLM Encoder] → 语言 Token ──────┼→ [LLM 主干 Transformer] → 动作 Token
                                                │
    本体感知 → [状态编码器] → 状态 Token S ─────┘
                                                ↓
                                   [动作预测头: D 个 256 类分类头]
                                                ↓
                                         离散动作索引 → [动作还原] → 连续动作 at


<div>

### 1.5 动作空间离散化

VLA 的一个核心挑战是：**如何将连续的动作空间与基于 Token 的 LLM 架构融合？**

RT-2 提出的解决方案：将连续动作离散化到 256 个 Token 桶中。假设动作 a ∈ \[a~min~, a~max~\]：


a~discrete~ = Clamp( ⌊(a − a~min~) / (a~max~ − a~min~) × 255⌋, 0, 255 )

对于一个 D 维动作向量（如 7-DOF 机械臂），每个维度独立离散化：


**a**~discrete~ = \[a~1~^disc^, a~2~^disc^, \..., a~D~^disc^\]

推理时还原为连续值：


a~continuous~ = a~min~ + a~discrete~ / 255 × (a~max~ − a~min~)

  分桶方式       描述                      优点             缺点
  -------------- ------------------------- ---------------- ------------------
  **均匀分桶**   动作范围均匀划分 256 桶   简单直观         稀疏区间浪费
  **数据驱动**   根据数据分布动态定边界    更精确覆盖分布   需预分析数据分布
  **对数分桶**   取对数后均匀分桶          覆盖大范围动作   不适合均匀分布

> **🔑 离散化的核心优势**
>
> 1.  **与 LLM 统一**：动作 Token 直接复用语言 Token 的建模框架
> 2.  **语义化**：不同桶可能学到有意义的语义（如"左转"、"右转"）
> 3.  **多模态建模**：softmax 天然支持多峰动作分布


<div>

### 1.6 动作预测头设计

动作预测头是 VLA 架构中负责将 LLM 表示转换为具体动作输出的模块。三种常见设计：

  类型             原理                         公式                           代表
  ---------------- ---------------------------- ------------------------------ ----------------------
  **离散分类头**   每维度 256 类 softmax 分类   Softmax(**W**~i~ · **h**~L~)   RT-2, OpenVLA
  **连续回归头**   MLP 直接回归连续值           **w**~i~^T^**h**~L~ + b~i~     ACT, OpenVLA(混合)
  **扩散动作头**   去噪过程逐步生成动作         ε~θ~(a~t~, t, c)               π0, Diffusion Policy

以 RT-2 为例，其动作预测头输出格式为：

    [STOP Token] [Action Dim 1] [Action Dim 2] ... [Action Dim D]
       1 Token      256 类分类      256 类分类          256 类分类

每个动作维度对应一个独立的 256 分类头，最终选择概率最高的桶作为该维度的动作预测。

</div>

<div>

### 1.7 VLA 训练三阶段

VLA 训练通常分为三个阶段，每个阶段的数据和目标各不相同：

  阶段           数据                                   目标                 参数策略
  -------------- -------------------------------------- -------------------- -----------------------------------
  **预训练**     图像-文本对（LAION-2B）                视觉-语言基础对齐    对齐模块训练，编码器冻结
  **指令微调**   视觉-语言指令数据                      指令跟随、任务理解   对齐模块 + 指令遵循训练
  **动作微调**   机器人演示轨迹（图像+指令+动作标签）   动作预测能力注入     视觉微调 + LLM(LoRA) + 动作头全量

    ┌─────────────────────────────────────────────────────┐
    │                    预训练阶段                        │
    │  数据: 图像-文本对（LAION-2B 等）                    │
    │  目标: 视觉-语言基础对齐                             │
    │  参数: 对齐模块训练，VL 编码器冻结                    │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │                   指令微调阶段                        │
    │  数据: 视觉-语言指令数据（LLaVA-Instruct 等）         │
    │  目标: 指令跟随、任务理解                             │
    │  参数: 对齐模块 + 指令遵循训练                        │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │                   动作微调阶段                        │
    │  数据: 机器人演示轨迹（图像 + 指令 + 动作标签）        │
    │  目标: 动作预测能力注入                              │
    │  参数: 视觉编码器微调 + LLM(LoRA) + 动作预测头        │
    └─────────────────────────────────────────────────────┘
                              ↓
                          完整 VLA 模型

> **⚠️ ⚠️ 动作微调是关键**
>
> 前两阶段本质上和训练一个 VLM 没有区别。第 3 阶段才是 VLA 与 VLM 的**本质分水岭**------模型第一次见到"图像 + 指令 → 动作"的配对数据。OpenVLA 采用的策略是：**视觉编码器全量微调 + LLM 使用 LoRA + 动作头全量训练**。

</div>

<div>

### 1.8 连续动作 vs 离散动作

VLA 在处理动作空间时面临一个根本性选择：

  维度               连续动作         离散动作
  ------------------ ---------------- ---------------------
  **动作精度**       高（任意精度）   受桶数量限制
  **动作空间覆盖**   自然覆盖         可能存在稀疏区间
  **多峰分布**       需要混合模型     softmax 天然支持
  **与 LLM 融合**    需要额外设计     直接复用 Token 框架
  **代表工作**       ACT, π0          RT-2, OpenVLA

> **📈 行业趋势**
>
> 2025 年后，**扩散式连续动作生成**（π0 路线）正在成为主流。它兼具连续动作的高精度和概率建模的多峰能力，是本课程第 5 章的重点。

</div>

<div>

### 1.9 实时性要求与推理优化

机器人控制对实时性有严格要求，VLA 推理延迟是实际部署中的关键瓶颈：

  应用场景   控制频率     延迟容忍度
  ---------- ------------ ------------
  工业装配   500Hz-1kHz   \<2ms
  动态操作   50-100Hz     \<20ms
  日常家务   10-30Hz      \<100ms
  任务规划   1-5Hz        \<1s

#### 推理延迟构成

  阶段           OpenVLA (7B)        RT-2 (55B)
  -------------- ------------------- ---------------------
  视觉编码       \~10ms              \~30ms
  LLM 前向传播   \~30ms (INT4)       \~500ms (FP16)
  **总延迟**     **\~40ms (25Hz)**   **\~530ms (\~2Hz)**

#### 五大优化手段

1.  **模型量化**：INT8 降 30-40% 延迟，INT4 降 50-60%，可运行于消费级 GPU
2.  **模型蒸馏**：大模型 Teacher → 小模型 Student，OpenVLA-3B 比 7B 快 2 倍
3.  **推测解码**：小模型预测动作，大模型验证
4.  **异步执行**：控制与推理解耦，MPC 与 VLA 协同
5.  **动作缓存**：静态场景缓存预测，避免重复推理

</div>

<div>

### 1.10 本课程选型理由

本课程围绕三款 VLA 模型展开，选择逻辑如下：

  模型          参数   为什么选它                             在哪练
  ------------- ------ -------------------------------------- --------------------
  **OpenVLA**   7B     开源标杆，文档完善，学术界基准         第 3 章 · 仿真推理
  **SmolVLA**   450M   极致轻量，L40 单卡训练，成本最低       第 4 章 · 云端训练
  **π0**        \~3B   扩散 VLA 代表，SOTA 性能，理解新范式   第 5 章 · L40 训练

> **🔑 学习路线**
>
> 先用 OpenVLA 理解 VLA 推理（第 3 章）→ 用 SmolVLA 上手训练（第 4 章）→ 用 π0 接触前沿扩散范式（第 5 章）→ 云端训练实战（第 6 章）→ 真机部署（第 7 章）。**从推理到训练，从仿真到真机，从经典到前沿。**

</div>

<div>

### 1.11 VLA 的能力边界


⚠️ 当前 VLA 还做不到的

-   **长时序规划**：难以处理超过 20 步的复杂任务链
-   **精确力控**：插拔 USB、拧螺丝等需要毫牛级力觉的任务
-   **新物体泛化**：没见过形状的物体仍会失败，域外泛化是开放问题
-   **高速实时控制**：7B+ 模型推理 \>1 秒，不适合 50Hz+ 的控制回路
-   **安全性保证**：模型可能输出危险的超界动作，需要工程防护


<div>

### 1.12 动手实验 1：动作离散化

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

### 1.13 动手实验 2：Token 编码解码

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

> **🔄 从 §1 的 VLA 理论出发** 你已经理解了 VLA 的三模态架构和动作离散化原理。现在在把这些概念落地的第一步------搭建仿真沙盒。MuJoCo 提供物理引擎，LIBERO 提供标准 VLA 评测任务，两者结合就是你的 VLA 实验室。

VLA 基础与架构
--------------

视觉-语言-动作模型的核心概念、技术演进与系统架构。

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

</div>

</div>

</div>


</div>


</div>


</div>

</div>
