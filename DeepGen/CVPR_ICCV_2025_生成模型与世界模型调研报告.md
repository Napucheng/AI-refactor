# CVPR 2025 / ICCV 2025 视觉领域生成模型与世界模型 深度调研报告

> 调研时间：2026-04-13 | 覆盖会议：CVPR 2025, ICCV 2025 | 方向：生成模型 & 世界模型

---

## 一、生成模型（Generative Models）—— 5 篇代表性论文

### 1. VGGT: Visual Geometry Grounded Transformer ⭐ CVPR 2025 最佳论文

- **ArXiv**: [2503.11651](https://arxiv.org/abs/2503.11651)
- **作者**: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny (Oxford VGG & Meta AI)
- **研究方向**: 3D 场景理解与生成 / 多视图几何的端到端学习

**摘要**:
本文提出了 VGGT，一种前馈神经网络，能直接推断场景的所有关键 3D 属性，包括相机参数、点图、深度图和 3D 点轨迹。该架构使用交替注意力机制，在分析单帧和整合所有图像信息之间切换，平衡局部细节与全局一致性。VGGT 将 3D 重建从传统的迭代优化问题重新定义为学习预测问题。

**为什么值得深挖**:
- 在 RealEstate10K 上实现 85.3 AUC@30（仅 0.2 秒），比 DUSt3R 快 35-50 倍
- 代表了从优化范式向学习范式的根本性转变
- 对 AR、机器人导航、自动驾驶的实时 3D 理解具有重大意义

---

### 2. DDT-LLaMA: Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens ⭐ CVPR 2025 Best Student Paper Honorable Mention (Oral)

- **ArXiv**: [2504.14666](https://arxiv.org/abs/2504.14666)
- **作者**: Kaihang Pan, Wang Lin, Zhongqi Yue, Tenglong Ao, Liyu Jia, Wei Zhao, Juncheng Li, Siliang Tang, Hanwang Zhang
- **研究方向**: 多模态生成统一 / 离散视觉 Token 化

**摘要**:
现有 MLLM 使用空间排列的视觉 token，但空间 token 缺乏语言的递归结构，形成 LLM 难以掌握的"不可能语言"。本文提出 DDT（Discrete Diffusion Timestep）tokenization，利用扩散时间步学习具有递归结构的离散视觉 token。每个 token 代表在特定噪声级别下丢失的图像信息，token 之间自然形成层次递进关系，使 LLM 能有效整合自回归推理和扩散生成能力。

**为什么值得深挖**:
- 提出了全新的视觉 tokenization 范式，将扩散过程编码为递归视觉语言
- 在多模态理解和生成的统一框架中取得 SOTA
- 解决了长期以来视觉 token 不适合语言模型的核心难题

---

### 3. Parallelized Autoregressive Visual Generation ⭐ CVPR 2025

- **ArXiv**: [2412.15119](https://arxiv.org/abs/2412.15119)
- **作者**: Xinhao Li, Yu Liu, et al.
- **研究方向**: 自回归视觉生成加速 / 并行推理

**摘要**:
自回归模型在视觉生成中表现出色，但逐 token 预测导致推理速度极慢。本文提出一种简单的并行自回归生成方法：核心观察是，弱依赖的远距离 token 可以并行生成，而强依赖的相邻 token 需要顺序生成。该方法无需修改模型架构或 tokenizer，即可无缝集成到标准自回归模型中。在 ImageNet 和 UCF-101 上实现了 3.6× 加速（同等质量）和最高 9.5× 加速（轻微质量损失）。

**为什么值得深挖**:
- 为自回归视觉模型的实际部署提供了关键效率解决方案
- 方法简洁优雅，无需改架构，通用性强
- 是自回归 vs 扩散范式竞争中的重要技术补充

---

### 4. GigaTok: Scaling Visual Tokenizers to 3 Billion Parameters for Autoregressive Image Generation ⭐ ICCV 2025 Highlight

- **ArXiv**: [2504.08736](https://arxiv.org/abs/2504.08736)
- **作者**: Cheng-Hao Tu, et al. (SilentView)
- **研究方向**: 视觉 Tokenizer 扩展 / 自回归图像生成

**摘要**:
在 AR 图像生成中，视觉 tokenizer 将图像压缩为紧凑的离散潜在 token。虽然扩展 tokenizer 提升了重建质量，但通常损害下游生成质量——即"重建 vs 生成困境"。本文提出 GigaTok，首次在扩展 tokenizer 时同时改善图像重建、生成和表征学习。核心是"语义正则化"（Semantic Regularization），将 tokenizer 特征与预训练视觉编码器的语义一致特征对齐，防止潜在空间过度复杂化。2.9B 参数的 GigaTok 在 ImageNet 256×256 上配合 1.4B AR 模型取得 SOTA。

**为什么值得深挖**:
- 首次系统解决了 AR 视觉模型中 tokenizer 扩展的困境
- 揭示了潜在空间复杂度与生成质量之间的深层关系
- 对未来大规模视觉生成系统的设计具有指导意义

---

### 5. Neural Inverse Rendering from Propagating Light ⭐ CVPR 2025 Best Student Paper

- **ArXiv**: 暂未在 arXiv 公开（CVPR 官方页面可获取）
- **作者**: Anagh Malik, Benjamin Attal, Andrew Xie, Matthew O'Toole, David B. Lindell (U of Toronto, CMU)
- **研究方向**: 物理驱动的神经渲染 / 光传输逆向建模

**摘要**:
传统 LiDAR 系统只利用直射光信息，丢弃了包含丰富场景信息的间接光（多次散射光）。本文提出时间分辨辐射缓存（Time-resolved Radiance Cache），一种神经网络学习存储和查询光在场景中传播的信息。该方法使 LiDAR 不仅能理解直射光，还能解码间接光，用于场景重建。系统可合成新视角下的光传播视频、自动分离直射和间接照明、以及新光源重照明。

**为什么值得深挖**:
- 将物理（光传输方程）与神经网络深度融合的典范
- 开创了利用间接光信息进行高质量 3D 重建的新路径
- 对自动驾驶感知、复杂光照场景理解有重要应用价值

---

## 二、世界模型（World Models）—— 5 篇代表性论文

### 1. Navigation World Models ⭐ CVPR 2025 Best Paper Honorable Mention

- **ArXiv**: [2412.03572](https://arxiv.org/abs/2412.03572)
- **作者**: Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun (Meta FAIR, NYU, BAIR)
- **研究方向**: 具身智能世界模型 / 导航规划

**摘要**:
导航是具身智能体的基本技能。本文提出 Navigation World Model (NWM)，一种可控视频生成模型，基于过去的观测和导航动作预测未来的视觉观测。NWM 采用条件扩散 Transformer (CDiT)，在多种第一人称视角视频（人类和机器人）上训练，扩展到 10 亿参数。在熟悉环境中，NWM 可以通过模拟来规划导航轨迹。与固定行为的有监督导航策略不同，NWM 能在规划中动态融入约束。更重要的是，NWM 能仅凭单张输入图像在陌生环境中想象出导航路径。

**为什么值得深挖**:
- Yann LeCun 团队的核心成果，代表了 JEPA 世界模型路线的重要实践
- 从"反应式导航"转向"预测式导航"的范式变革
- 在已知和未知环境中均展示了强大的泛化能力（引用 198 次，影响力极高）

---

### 2. How Far is Video Generation from World Model: A Physical Law Perspective ⭐ 被引 140 次

- **ArXiv**: [2411.02385](https://arxiv.org/abs/2411.02385)
- **作者**: Kang et al.
- **研究方向**: 视频生成 vs 世界模型的评估基准 / 物理规律学习

**摘要**:
OpenAI Sora 展示了视频生成发展为世界模型的潜力。但视频生成模型能否仅从视觉数据中发现基本物理定律？本文从物理定律视角系统评估了这一问题。作者构建了 2D 模拟实验平台，生成由经典力学定律（物体运动、碰撞）确定性支配的视频。通过三个维度评估：分布内、分布外和组合泛化。实验表明扩散视频生成模型在分布内完美泛化，组合泛化有可量化的缩放行为，但在分布外场景中失败。这揭示了当前视频生成模型与真正世界模型之间的关键差距。

**为什么值得深挖**:
- 第一次以严格的物理实验框架量化了视频生成模型的"世界模型"能力
- 明确指出当前模型的根本局限：学到关联而非因果
- 为世界模型评估提供了可复现的基准和方法论
- 引用量极高（140+），在领域内产生了广泛影响

---

### 3. Vid2World: Crafting Video Diffusion Models to Interactive World Models

- **ArXiv**: [2505.14357](https://arxiv.org/abs/2505.14357)
- **作者**: Huang, Wu, et al.
- **研究方向**: 视频扩散模型 → 交互式世界模型的转换

**摘要**:
世界模型需要预测基于过去观测和动作序列的未来转换，在增强智能体数据效率方面展示了巨大潜力。本文提出 Vid2World，一种将预训练视频扩散模型（如 Sora 类模型）改造为交互式世界模型（如 Genie 类模型）的通用方法。核心是通过"因果化"（Causalization）改造预训练模型的架构和训练目标，使其具备动作条件控制能力。作者在广泛预训练的 1.4B 参数视频扩散模型上进行了验证，成功迁移到多种交互领域。

**为什么值得深挖**:
- 提出了将大规模视频生成模型复用为世界模型的实用路径
- 避免了从头训练世界模型的巨大成本
- 在"视频生成 → 世界模型"的桥梁搭建上具有方法论意义

---

### 4. Long-Context State-Space Video World Models ⭐ ICCV 2025（引用 37 次）

- **ArXiv**: 待确认（ICCV 2025 Open Access: openaccess.thecvf.com/content/ICCV2025）
- **作者**: Po et al.
- **研究方向**: 长序列视频世界模型 / 状态空间模型

**摘要**:
视频扩散模型近来通过基于动作条件的自回归帧预测展示了世界建模的前景，但受限于有限的上下文窗口，难以记忆长期环境动态。本文提出长上下文状态空间视频世界模型，利用状态空间模型（SSM）架构来建模长范围时间依赖关系，使模型能够在更长的视频序列上学习环境动态，从而产生更一致和物理合理的未来预测。

**为什么值得深挖**:
- 解决了视频世界模型中的长上下文记忆难题
- 状态空间模型（如 Mamba）与视频生成的结合是前沿方向
- 对需要长期规划和记忆的具身应用至关重要

---

### 5. Learning 4D Embodied World Models (TesserAct) ⭐ ICCV 2025（引用 1 次）

- **来源**: ICCV 2025 Open Access (openaccess.thecvf.com/content/ICCV2025/papers/Zhen_Learning_4D_Embodied_World_Models_ICCV_2025_paper.pdf)
- **作者**: Zhen et al.
- **研究方向**: 4D 具身世界模型 / 多模态 3D+时间建模

**摘要**:
本文提出 TesserAct，一种 4D 具身世界模型，接收输入图像和文本指令，生成 RGB、深度和法线视频，重建完整的 4D 场景表示。该模型不仅预测 2D 帧的演变，还预测完整的 3D 场景随时间的动态变化，使智能体能够进行多模态的空间理解和时间推理。

**为什么值得深挖**:
- 将世界模型从 2D 视频预测提升到完整的 4D（3D + 时间）场景理解
- 多模态输出（RGB + 深度 + 法线）提供了比纯视频更丰富的场景表征
- 对机器人的空间推理和操作规划具有直接应用价值

---

## 三、关键趋势总结

### 生成模型方向
1. **自回归复兴**: AR 模型在视觉生成领域重新崛起（GigaTok, Parallelized AR），与扩散模型形成竞争态势
2. **统一理解与生成**: DDT-LLaMA 代表了在单一框架内统一视觉理解和生成的新范式
3. **效率至关重要**: 并行推理（Parallelized AR）和 tokenizer 优化（GigaTok）成为研究热点
4. **物理信息融合**: 从纯数据驱动转向物理+学习的混合方法（Neural Inverse Rendering, VGGT）

### 世界模型方向
1. **从视频生成到世界模型**: 明确的研究路线，但评估和度量仍是挑战
2. **具身导航是核心应用场景**: Navigation World Models, TesserAct 都聚焦于具身智能
3. **物理规律学习是关键瓶颈**: How Far is Video Generation from World Model 揭示了当前模型的本质局限
4. **架构创新**: 状态空间模型（SSM）、条件扩散 Transformer 等新架构不断涌现

---

## 参考来源

1. [CVPR 2025 Best Papers and Best Demos](https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos)
2. [CVPR 2025 Top Papers: Award Winners and Notable Research (BasicAI)](https://www.basic.ai/blog-post/cvpr-2025-top-papers-award-winners-and-notable-research)
3. [CVPR 2025 Accepted Papers](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)
4. [ICCV 2025 Papers](https://openaccess.thecvf.com/ICCV2025)
5. [VGGT: arXiv 2503.11651](https://arxiv.org/abs/2503.11651)
6. [DDT-LLaMA: arXiv 2504.14666](https://arxiv.org/abs/2504.14666)
7. [Parallelized AR: arXiv 2412.15119](https://arxiv.org/abs/2412.15119)
8. [GigaTok: arXiv 2504.08736](https://arxiv.org/abs/2504.08736)
9. [Navigation World Models: arXiv 2412.03572](https://arxiv.org/abs/2412.03572)
10. [How Far is Video Generation from World Model: arXiv 2411.02385](https://arxiv.org/abs/2411.02385)
11. [Vid2World: arXiv 2505.14357](https://arxiv.org/abs/2505.14357)
12. [CVPR 2025 World Model Tutorial](https://world-model-tutorial.github.io)
