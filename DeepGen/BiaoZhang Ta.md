很好，这一步你已经从“做个项目”升级到**选对技术路线 + 讲出研究故事**了。导师真正看重的是：
👉 *你能不能把一个方向讲清楚 + 做出一个有 insight 的小系统*

我分三块讲：**3D生成 / 世界模型 / VLA**
每一块给你：**代表模型 → 学什么 → 能做什么项目（重点）**

------

# 一、3D生成（视觉→空间理解）

## 🔥 代表模型（按重要性排序）

### 1️⃣ NeRF 系列（基础必懂）

- NeRF（原始）
- Instant-NGP（工程优化版）
- NeRFStudio（强烈推荐）

👉 学什么：

- 连续场表示（implicit representation）
- 体渲染（volume rendering）
- 多视角一致性

------

### 2️⃣ 3D Gaussian Splatting（当前最火）

- 论文：3D Gaussian Splatting (SIGGRAPH 2023)
- 各种实时版本（SOTA）

👉 学什么：

- 显式表示（点 + 高斯）
- 实时渲染
- optimization-based training

------

### 3️⃣ 文本到3D（更前沿）

- DreamFusion（Google）
- Magic3D
- Zero-1-to-3

👉 学什么：

- diffusion + 3D
- score distillation

------

## 💡 你可以做什么（两周可落地）

### ✅ 项目 1（强烈推荐）

## 👉 “从单张图像生成多视角（Pseudo-3D）”

做法：

- 输入：1张图片
- 输出：不同视角图

方法：

- 用 diffusion / NeRF 思想
- 或直接用已有模型 + 自己改

👉 为什么导师会喜欢：

- 这是 2D→3D 的核心问题
- 非常有扩展空间

------

------

# 二、世界模型（最推荐你做）

## 🔥 代表模型

### 1️⃣ DreamerV3（最重要）

👉 DeepMind 当前最强 world model

特点：

- latent dynamics（RSSM）
- 可以做 RL

------

### 2️⃣ PlaNet / RSSM（基础）

👉 Dreamer 的前身

------

### 3️⃣ Video Diffusion

- Imagen Video
- Video LDM

------

### 4️⃣ Transformer World Model

- Genie（Google）
- VideoGPT

------

## 💡 你可以做什么（非常关键）

### ✅ 项目 1（最推荐）

## 👉 “学习一个可预测未来的世界模型”

任务：

- 输入前几帧
- 预测未来

但重点不是这个，而是👇

------

### ⭐ 升级点（导师会很喜欢）

## 👉 加一个“决策能力”

例如：

### 👉 任务升级：

- 给定目标（球要移动到某位置）
- 模型预测未来并选择动作

------

👉 你就从：

```
video prediction
```

变成：

```
latent planning
```

------

### ✅ 项目 2（更 research）

## 👉 “latent space 可解释性分析”

你做：

- 可视化 z_t
- 看是否学到：
  - 速度
  - 位置

👉 这是论文级别思路

------

# 三、VLA（Vision-Language-Action）

## 🔥 代表模型

### 1️⃣ RT-2（Google）

- vision + language → action

------

### 2️⃣ OpenVLA（开源）

👉 你可以真正跑的

------

### 3️⃣ PaLM-E

- embodied multimodal model

------

### 4️⃣ Diffusion Policy（强相关）

👉 action generation

------

## 💡 你可以做什么

### ✅ 项目 1（推荐）

## 👉 “语言控制的轨迹生成”

任务：
输入：

- “go to the red goal avoiding obstacle”

输出：

- 一条路径

方法：

- diffusion / transformer

------

### ✅ 项目 2（更简单）

## 👉 “语言 → reward → planning”

做法：

- 用 LLM 解析指令
- 转成 reward
- 再规划路径

------

👉 这是 mini VLA

------

## ✅ 导师喜欢的

你要做到：

### 1️⃣ 有“问题意识”

比如：

- 为什么 deterministic world model 不行？
- 为什么 diffusion 适合 trajectory？

------

### 2️⃣ 有“对比实验”

哪怕很简单：

| 方法     | 结果 |
| -------- | ---- |
| baseline | 模糊 |
| yours    | 清晰 |

------

# 六、对应 GitHub（你可以直接用）

我帮你配好：

------

## 3D

- https://github.com/nerfstudio-project/nerfstudio
- https://github.com/graphdeco-inria/gaussian-splatting

------

## World Model

- https://github.com/danijar/dreamerv3
- https://github.com/google-research/video_prediction

------

## Diffusion / Planning

- https://github.com/jannerm/diffuser
- https://github.com/real-stanford/diffusion_policy

------

## VLA

- https://github.com/openvla/openvla
- https://github.com/google-deepmind/rt-2 (部分开源)

------

# 七、给你一个现实建议（很重要）

两周时间：

👉 不要贪三个方向

------

## 最优策略：

### 主线：

👉 World Model（70%精力）

### 副线：

👉 Diffusion（30%融合进去）

------

- **Autoregressive Models**
  - **RT系列(Robotic Transformers)**:
    - **RT-1** ([paper](https://arxiv.org/abs/2212.06817))
    - **RT-2** ([page](https://robotics-transformer2.github.io/) | [paper](https://arxiv.org/abs/2307.15818), Google Deepmind, 2023.7)：55B
    - **RT-Trajectory** ([paper](https://arxiv.org/pdf/2311.01977), Google Deepmind, UCSD, 斯坦福 2023.11)
    - **AUTORT** ([paper](https://arxiv.org/abs/2401.12963), Google Deepmind, 2024.1)
  - **RoboFlamingo** ([paper](https://arxiv.org/abs/2311.01378) | [code](https://github.com/roboflamingo), 字节、清华, 2024.2)
  - **OpenVLA** ([paper](https://arxiv.org/pdf/2406.09246) | [code](https://github.com/openvla), Stanford, 2024.6): 7B
  - **TinyVLA** ([paper](https://arxiv.org/abs/2409.12514), 上海大学, 2024.11)
  - **TraceVLA** ([paper](https://arxiv.org/pdf/2412.10345) | [code](https://lumina-embodied.ai/blog/eai-guide), 微软，2024.12)
- **Diffusion Models for Action Head:**
  - **Octo** ([paper](https://arxiv.org/pdf/2405.12213) | [code](https://octo-models.github.io/), 斯坦福，伯克利, 2024.5): Octo-base (93M)
  - **π0** ([paper](https://arxiv.org/pdf/2410.24164) | [code](https://github.com/Physical-Intelligence/openpi), 斯坦福, physical intelligence, ) : 3.3B; flow-based diffusion VLA; PaliGemma (3B VLM);
  - **CogACT** ([paper](https://arxiv.org/pdf/2411.19650) | [code](https://github.com/microsoft/CogACT.git), 清华，MSRA, 2024.11): 7B
  - **Diffusion-VLA** ([paper](https://arxiv.org/abs/2412.03293) | [code](https://arxiv.org/pdf/2410.07864), 华东师范，上海大学，美的, 2024.12)
- **3D Vision:**
  - **3D-VLA** ([paper](https://arxiv.org/pdf/2403.09631) | [code](https://github.com/UMass-Foundation-Model/3D-VLA/tree/main), UMass, 2024.3): 3D-based LLM
  - **SpatialVLA** ([paper](https://arxiv.org/pdf/2501.15830) | [code](https://github.com/SpatialVLA/SpatialVLA) , 上海AI Lab, 2025.1): Adaptive Action Grid
- **VLA-related:**
  - **FAST (π0)** ([paper](https://arxiv.org/pdf/2410.24164), [code](https://github.com/Physical-Intelligence/openpi.git), 斯坦福，伯克利, physical intelligence, 2025.1): autoregressive VLA
  - **RLDG** ([paper](https://generalist-distillation.github.io/static/high_performance_generalist.pdf) | [code](https://arxiv.org/abs/2410.01971), 伯克利, 2024.12 ): 强化学习(RL)生成高质量的训练数据进行微调
  - **BYO-VLA** ([paper](https://arxiv.org/abs/2410.01971) | [code](https://github.com/irom-princeton/byovla), 普林斯顿大学, 2024.10): 运行时图像干预，有效降低VLA模型对任务无关视觉干扰的敏感度



VAE/GAN/Flow matching/Diffusion

[What are Diffusion Models? | Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

diffusion包含：

- Classifier Guided Diffusion
- Stable Diffusion SD1.5/SDXL/SD3.0/DDIM
- SDE/ODE

- [arXiv](https://arxiv.org/abs/2112.10752)
- [GitHub](https://github.com/CompVis/stable-diffusion)

DALL·E2

sora2



NERF

3DGS

- rendering equation（体渲染公式）

- differentiable rendering

[Pointcept/PointTransformer CVPR'24 Oral\] Official repository of Point Transformer V3 (PTv3)](https://github.com/Pointcept/PointTransformerV3)



[(40 封私信 / 80 条消息) 3D生成相关论文-2024 - 知乎](https://zhuanlan.zhihu.com/p/700895749)

[(40 封私信 / 80 条消息) Diffusion Model for 2D/3D Generation 相关论文分类 - 知乎](https://zhuanlan.zhihu.com/p/617510702)

关键趋势：

- text → 3D
- image → 3D



- DreamFusion（Google）
- Magic3D
- Zero-1-to-3

👉 必看方向：

- DreamFusion（NeRF + diffusion）
- Score Distillation Sampling（SDS）



[视频生成的扩散模型 |小洛格 --- Diffusion Models for Video Generation | Lil'Log](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)

[Video Understanding with Large Language Models: A Survey](https://arxiv.org/pdf/2312.17432)

[cwchenwang/awesome-4D-generation：关于 4D 生成的论文列表。 --- cwchenwang/awesome-4d-generation: List of papers on 4D Generation.](https://github.com/cwchenwang/awesome-4d-generation)



[Spatial VLA](https://arxiv.org/pdf/2501.15830)

diffusion world model