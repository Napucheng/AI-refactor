# Embodied AI Foresight

取源自Lumina具身智能开源社区

## Conference

Science Robotics, TRO, IJRR, JFR, RSS, IROS, ICRA, ICCV, ECCV, ICML, CVPR, NIPS, ICLR, AAAI, ACL

## Community

[宇树具身智能社群](https://www.unifolm.com/#/)

[Simulately-关于仿真](https://simulately.wiki/)

[Huggingface LeroBot](https://github.com/huggingface/lerobot)

- 基于 **PyTorch** 的**开源机器人学习全栈框架**，它相当于机器人领域的 **Transformers**：提供统一接口、预训练模型、标准化数据集，让你快速在仿真或真实机器人上跑模仿学习、强化学习、VLA模型。

## Skills/Models/Algorithm

[3D点云降采样](https://zhuanlan.zhihu.com/p/558683732?utm_campaign=shareopn&utm_medium=social&utm_psn=1772067996070236160&utm_source=wechat_session)

[手眼标定](https://github.com/fishros/handeye-calib)

### Vision Fundamental Models

- CLIP: [website](https://github.com/openai/CLIP), 来自OpenAI的研究, 最基本的应用是可以计算图像与语言描述的相似度, 中间层的视觉特征对各种下游应用非常有帮助。
- DINO: [DINO repo](https://github.com/facebookresearch/dino), [DINO-v2 repo](https://github.com/facebookresearch/dinov2), 来自Meta的研究, 可以提供图像的高层视觉特征, 对corresponding之类的信息提取非常有帮助, 比如不同个体之间的鼻子都有类似的几何特征, 这个时候不同图像中关于不同鼻子的视觉特征值可能是近似的。
- SAM: [website](https://segment-anything.com/), 来自Meta的研究, 可以基于提示点或者框, 对图像的物体进行分割。
- SAM2: [website](https://ai.meta.com/sam2/), 来自Meta的研究, SAM的升级版, 可以在视频层面持续对物体进行分割追踪。
- Grounding-DINO: [repo](https://github.com/IDEA-Research/GroundingDINO), [在线尝试](https://deepdataspace.com/playground/grounding_dino), **这个DINO与上面Meta的DINO没有关系**, 是一个由IDEA研究院(做了很多不错开源项目的机构)开发集成的图像目标检测的框架, 很多时候需要对目标物体进行检测的时候可以考虑使用。
- OmDet-Turbo: [repo](https://github.com/om-ai-lab/OmDet), 一个由OmAI Lab开源的研究, 提供OVD（开放词表目标检测）能力, 优点在于推理速度非常快（100+FPS）, 适合需要高FPS的自定义目标物体检测场景。
- Grounded-SAM: [repo](https://github.com/IDEA-Research/Grounded-SAM-2), 比Grounding-DINO多了一个分割功能, 也就是支持检测后分割, 也有很多下游应用, 具体可以翻一下README。
- FoundationPose: [website](https://github.com/NVlabs/FoundationPose), 来自Nvidia的研究, 物体姿态追踪模型。
- Stable Diffusion: [repo](https://github.com/CompVis/stable-diffusion), [website](https://ommer-lab.com/research/latent-diffusion-models/), 22年的文生图模型, 现在虽然不是SOTA了, 但是依然可以作为不错的应用, 例如中间层特征支持下游应用、生成Goal Image (目标状态) 等等。
- Depth Anything (v1 & v2): [repo](https://github.com/LiheYoung/Depth-Anything), [repo](https://github.com/DepthAnything/Depth-Anything-V2), 港大和字节的研究工作, 单目深度估计模型。
- Point Transformer (v3): [repo](https://github.com/Pointcept/PointTransformerV3), 点云特征提取的工作。
- RDT-1B: [website](https://rdt-robotics.github.io/rdt-robotics/), 清华朱军老师团队的工作, 机器人双臂操作的基础模型, 具有强大的few-shot能力。
- SigLIP: [huggingface](https://huggingface.co/docs/transformers/en/model_doc/siglip), 类似CLIP。

### Robot 

#### MPC

模型预测控制（MPC）是一种先进的控制策略，利用系统的显式动态模型预测有限时间范围内的未来行为。每个控制周期，MPC 通过求解优化问题来确定控制输入，以优化指定的性能指标，同时满足输入和输出的约束条件。优化序列中的第一个控制输入应用于系统，在下一个时间步中，结合新的系统状态测量或估计，重复该过程。

- 入门推荐视频：

  - Model Predictive Control 模型预测控制,从公式到代码 - 华工机器人实验室: [bilibili](https://www.bilibili.com/video/BV1U54y1J7wh/?spm_id_from=333.999.0.0&vd_source=180b6da13847c26de9d19ac71e61c7fe); 仿真工程源码:[Gitee](https://gitee.com/clangwu/mpc_control.git) 这门课程适合作为从PID到MPC的入门课程，适合只了解PID控制原理，但不太清楚MPC原理的入门者；从公式原理推导，到CoppeliaSim（V-ERP）仿真教程以及MatLab代码编写，深入浅出。

- 经典工作：

  **理论基础**：

  - [Model predictive control: Theory and practice—A survey](https://www.sciencedirect.com/science/article/abs/pii/0005109889900022) ： 这篇全面的综述论文讨论了 MPC 的理论基础及其实践应用，为未来的研究奠定了基础。

  **非线性 MPC**：

  - [An Introduction to Nonlinear Model Predictive Control](https://pure.tue.nl/ws/files/3079152/555518.pdf#page=120) ： 提供了对非线性 MPC 的简明介绍，扩展了 MPC 在具有显著非线性系统中的应用。

  **显式 MPC**：

  - [The explicit linear quadratic regulator for constrained systems](https://www.sciencedirect.com/science/article/abs/pii/S0005109801001741) ： 讨论了显式 MPC 解的公式化，对于需要快速实时控制的系统至关重要。

  **鲁棒 MPC**：

  - [Predictive End-Effector Control of Manipulators on Moving Platforms Under Disturbance](https://ieeexplore.ieee.org/document/9425004) ： 使用时间序列分析预测基座运动并相应地转换期望轨迹，使得机械臂可以达到主动在扰动下的基座运动。是使用二次规划（QP）公式化模型预测控制（MPC）问题的经典之作。
  - [Min-max feedback model predictive control for constrained linear systems](https://ieeexplore.ieee.org/abstract/document/704989) ： 解决了 MPC 中的鲁棒性，提出了处理模型不确定性并确保在扰动下性能的方法。

  **基于学习的MPC**：

  - [Learning-Based Model Predictive Control for Safe Exploration](https://ieeexplore.ieee.org/abstract/document/8619572) ： 将机器学习与 MPC 相结合，代表了将数据驱动的模型和学习纳入控制的现代趋势。
  - [Confidence-Aware Object Capture for a Manipulator Subject to Floating-Base Disturbances](https://ieeexplore.ieee.org/document/10684104) ： 利用小波神经网络进行实时运动预测，并且引入置信度评价，实现短周期内最优轨迹规划，使得机械臂在扰动平面上抓取无人机（UAV）表现优异，具备良好的鲁棒性。

#### RL

- 强化学习的数学原理 - 西湖大学赵世钰: [bilibili](https://space.bilibili.com/2044042934/channel/collectiondetail?sid=748665) [GitHub](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) 这门课程作为强化学习的入门课程非常合适，适合只对机器学习略有了解，但没有了解过强化学习的初学者，可以了解强化学习的数学原理，其教材编写也十分用心。

下面列出三门比较受欢迎的深度强化学习相关的课程，这几门课互有overlap，时间长短和授课风格也各有不同，读者可以选择适合自己的课程进行学习。此外，深度强化学习的经典算法相关的文章也在必读清单：如[PPO](https://arxiv.org/abs/1707.06347), [SAC](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf), [TRPO](https://arxiv.org/abs/1502.05477), [A3C](https://arxiv.org/abs/1602.01783)等。

- The Foundations of Deep RL in 6 Lectures [YouTube](https://www.youtube.com/watch?v=2GwBez0D20A) 本门在线课程由在RL领域著名的Pieter Abbeel教授主讲，从MDP开始在六节课之内介绍了深度强化学习的主要知识。
- UC Berkeley CS285 深度强化学习: [website](https://rail.eecs.berkeley.edu/deeprlcourse/) | [YouTube](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps) 本课程的主讲老师是在RL领域著名的Berkeley的Sergey Levine教授，DRL领域许多著名的工作如SAC就出自他之手。Sergey在授课方面非常用心，本课程对DRL提供了非常详细的介绍。
- 李宏毅老师也有一套关于强化学习的课程: bilibili上课+刷蘑菇书巩固+gymnasium动手实践, 重点了解一下PPO。
  - 台湾大学李宏毅公开课: [bilibili](https://www.bilibili.com/video/BV1XP4y1d7Bk/?spm_id_from=333.337.search-card.all.click&vd_source=ab9cf5374617c2867aaea34af29b53c9)
  - EasyRL - 蘑菇书: [website](https://datawhalechina.github.io/easy-rl/#/), 基本是配套李宏毅老师的课程
  - 实践[gymnasium](https://gymnasium.farama.org/), 可以尝试一下把玩一下登月着陆等经典强化学习场景, 思考+动手, 观察阶段agent的表现并分析, 有助于深入理解强化学习

然而，深度强化学习的Reward Tuning和参数调整非常依赖于经验，建议读者在对深度强化学习有相关经验之后，可以自己尝试训练一个policy并在机器人上部署，体会其中的Sim-to-Real Gap。常用的仿真平台有[MuJoCo PlayGround](https://playground.mujoco.org/), [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html), [SAPIEN](https://sapien.ucsd.edu/), [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)等。

常用的Codebase有[legged-gym](https://github.com/leggedrobotics/legged_gym)(由ETH RSL开发，基于IsaacGym)等，也可以根据你想做的任务找到相近的codebase。

#### Imitation Learning

- 《模仿学习简洁教程》 - 南京大学LAMDA: [PDF](https://www.lamda.nju.edu.cn/xut/Imitation_Learning.pdf)
- Supervised Policy Learning for Real Robots, RSS 2024 Workshop 教程：真实机器人的监督策略学习, [bilibili](https://www.bilibili.com/video/BV1Fx4y1s7if/?buvid=XY415384A771A6C681C9BEB3817566ED57724&is_story_h5=false&mid=ORgXkVzTHaOKTsml0RX5Gw%3D%3D&plat_id=240&share_from=ugc&share_medium=android&share_plat=android&share_source=WEIXIN&share_tag=s_i&spmid=dt.space-dt.0.0&timestamp=1721464513&unique_k=Cqj5d9J&up_id=2185804&vd_source=ab9cf5374617c2867aaea34af29b53c9)

### LLM

为了促使机器人更好的规划, 现代具身智能工作常常利用大语言模型强大的信息处理能力与泛化能力进行规划。

- Robotics+LLM系列通过大语言模型控制机器人 [2]: [zhihu](https://zhuanlan.zhihu.com/p/668053911)
- Embodied Agent wiki: [website](https://en.wikipedia.org/wiki/Embodied_agent)
- Lilian Weng 个人博客 - AI Agent 系统综述 [5]: 中文: [website](https://mp.weixin.qq.com/s/Jb8HBbaKYXXxTSQOBsP5Wg) 英文: [website](https://lilianweng.github.io/posts/2023-06-23-agent/)
- 过去一系列工作常常仅使用LLM作为High-Level的策略生成器 用作High-Level 规划
  - 经典工作(1) PaLM-E: [Arxiv](https://arxiv.org/abs/2303.03378)
  - 经典工作(2) DO AS I CAN, NOT AS I SAY: [Arxiv](https://arxiv.org/abs/2204.01691)
  - 经典工作(3) Look Before You Leap: [Arxiv](https://arxiv.org/abs/2311.17842)
  - 经典工作(4) EmbodiedGPT: [Arxiv](https://arxiv.org/abs/2305.15021)
- 同时也有一些工作将High-Level的策略规划与 Low-Level的动作生成进行统一
  - 经典工作(1) RT-2: [Arxiv](https://arxiv.org/abs/2307.15818)
- 另一个代表性方向的工作将LLM与传统基于算法的Planner结合来做任务与移动规划
  - 经典工作(1) LLM+P: [Arxiv](https://arxiv.org/abs/2304.11477)
  - 经典工作(2) AutoTAMP: [Arxiv](https://arxiv.org/abs/2306.06531)
  - 经典工作(3) Text2Motion: [Arxiv](https://arxiv.org/abs/2303.12153)
- 利用LLM的code能力实现具身智能是一个有趣的想法
  - 经典工作(1) Code as Policy: [Arxiv](https://arxiv.org/abs/2209.07753)
  - 经典工作(2) Instruction2Act: [Arxiv](https://arxiv.org/abs/2305.11176)
- 有一些工作将三维视觉感知同LLM结合起来，共同促进具身智能规划
  - VoxPoser [Arxiv](https://arxiv.org/abs/2307.05973)
  - OmniManip [Arxiv](https://arxiv.org/abs/2501.03841)
- 还有一些工作试图把基于LLM的机器人规划扩展到多机器人协同合作的场景
  - 经典工作(1) RoCo: [Arxiv](https://arxiv.org/abs/2307.04738)
  - 经典工作(2) Scalable-Multi-Robot: [Arxiv](https://arxiv.org/abs/2309.15943)

### VLA

**Vision-Language-Action Models(VLA模型)** 是一种结合VLM(Vision-Language Model)与机器人控制的模型，旨在将预训练的VLM直接用于生成机器人动作(RT-2中定义)。和以往利用VLM做planning以及build from strach的方法不同，VLA无需重新设计新的架构，将动作转化为token，微调VLM。

**VLA的特点**：端到端，使用LLM/VLM backbone，加载预训练模型, etc.

目前的VLA可以从以下几个方面进行区分：模型结构&大小(如action head的设计, tokenize的方法如FAST)，预训练与微调策略和数据集，输入和输出(2D vs. 3D | TraceVLA输入visual trace)，不同的应用场景等。

**参考资料：**

- Blog: [具身智能Vision-Language-Action的思考](https://zhuanlan.zhihu.com/p/9880769870), [zhihu](https://www.zhihu.com/question/655570660/answer/87040917575)
- Survey: [A Survey on Vision-Language-Action Models for Embodied AI](https://arxiv.org/abs/2405.14093), 2024.11.28

**经典工作**：

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
- **Different Locomotion:**
  - **RDT-1B (双臂)** ([paper](https://arxiv.org/pdf/2410.07864) | [code](https://github.com/thu-ml/RoboticsDiffusionTransformer), 清华): 双臂控制的扩散模型
  - **QUAR-VLA (四足机器人)** ([paper](https://arxiv.org/pdf/2312.14457), 西湖大学，浙江大学, 2025.2.4)
  - **CoVLA (自动驾驶)** ([paper](https://arxiv.org/abs/2408.10845) | [page](https://turingmotors.github.io/covla-ad/), Turing, 2024.12)
  - **Mobility-VLA (导航)** ([paper](https://arxiv.org/pdf/2407.07775), Google Deepmind, 2024.7)
  - **NaVILA (腿式机器人导航)** ([paper](https://arxiv.org/pdf/2412.04453) | [code](https://navila-bot.github.io/), USCD, 2024.12)



### Advanced CV

#### 2D Vision

- 2D Vision 领域的经典代表作
  - CNN (卷积神经网络): [link](https://easyai.tech/ai-definition/cnn/)
  - ResNet (深度残差网络): [bilibili](https://www.bilibili.com/video/BV1P3411y7nn/?spm_id_from=333.1387.collection.video_card.click&vd_source=930ef08bfb2ff0db87ec20bf72a99855)
  - ViT (第一个将Transformer用在视觉领域): [bilibili](https://www.bilibili.com/video/BV15P4y137jb/?spm_id_from=333.1387.collection.video_card.click&vd_source=930ef08bfb2ff0db87ec20bf72a99855)
  - Swin Transformer (披着Transformer皮的CNN): [bilibili](https://www.bilibili.com/video/BV13L4y1475U/?spm_id_from=333.1387.collection.video_card.click&vd_source=930ef08bfb2ff0db87ec20bf72a99855)
  - 对比学习论文综述: [bilibili](https://www.bilibili.com/video/BV19S4y1M7hm/?spm_id_from=333.1387.collection.video_card.click&vd_source=930ef08bfb2ff0db87ec20bf72a99855)
- 以判别式模型为主的感知任务, 比如识别、分类、分割、检测等等, 看看即可, 现在继续刷点意义不大
- 生成式模型
  - 自回归综述: [PDF](https://arxiv.org/pdf/2411.05902)
  - 扩散模型综述: [PDF](https://arxiv.org/pdf/2209.00796)
  - 如果对扩散模型的理论推导感兴趣, 可以看苏剑林老师的博客 - 生成扩散模型漫谈(推导非常清楚): [link](https://kexue.fm/archives/9119)



#### 3D Vision

- 三维视觉导论 - Andreas Geiger: [website](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/computer-vision/) (重点关注课程作业)
- GAMES203 - 三维重建和理解: [bilibili](https://www.bilibili.com/video/BV1pw411d7aS/?share_source=copy_web&vd_source=0b7603f37af6d369a97df34525b149be)
- 三维生成的一些经典论文:
  - Diffusion Model for 2D/3D Generation 相关论文分类: [link](https://zhuanlan.zhihu.com/p/617510702)
  - 3D生成相关论文-2024: [link](https://zhuanlan.zhihu.com/p/700895749)



#### 4D Vision

- 视频理解
  - 开山之作: [bilibili](https://www.bilibili.com/video/BV1mq4y1x7RU/?spm_id_from=333.1387.collection.video_card.click&vd_source=930ef08bfb2ff0db87ec20bf72a99855)
  - 论文串讲: [bilibili](https://www.bilibili.com/video/BV1fL4y157yA?spm_id_from=333.788.videopod.sections&vd_source=930ef08bfb2ff0db87ec20bf72a99855)
  - LLM时代的视频理解综述: [PDF](https://arxiv.org/pdf/2312.17432)
- 4D 生成
  - 视频生成博客(英文): [link](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
  - 4D 生成的论文列表: [website](https://github.com/cwchenwang/awesome-4d-generation)



#### Visual Prompting

视觉提示是一种利用视觉输入引导大模型完成特定任务的方法，常用于具身智能领域。它通过提供示例图像、标注或视觉线索，让模型理解任务要求，而无需额外训练。例如，在机器人导航、操控等场景中，视觉提示可帮助模型适应新环境，提高泛化能力。相比传统方法，视觉提示具备更强的灵活性和可扩展性，使具身智能系统能够通过视觉信息快速适应复杂任务。

- 视觉提示综述：[paper](https://arxiv.org/abs/2409.15310)
- **PIVOT**, [page](https://pivot-prompt.github.io/): 通过将任务转化为迭代式视觉问答，实现在无需特定任务数据微调的情况下，zero-shot控制机器人系统和进行空间推理。
- **Set-of-Mark Visual Prompting for GPT-4V**: [page](https://som-gpt4v.github.io/)



#### Affordance Grounding

可供性锚定任务的目标是从图像中定位物体上能够与之交互的区域，充当了感知与行动之间的桥梁，是具身智能重要的一环。它不仅需要模型对物体及其局部结构的检测与识别，还需要模型理解物体与人或机器人之间的潜在互动关系。例如，在机器人抓取场景中，可供性锚定帮助模型寻找物体上最佳的抓取位置，从而确定最佳抓取角度。该方向通过整合计算机视觉，多模态大模型技术，能够在弱监督或零样本条件下实现对物体交互可能性的精确定位，提升机器人抓取、操作以及人机交互等任务的性能。

- **2D**
  - 跨视角学习可供性：**Cross-View-AG**, [paper](https://arxiv.org/pdf/2203.09905): 第三视角图像提供他者如何与物体交互的信息，帮助模型学习如何与第一视角图像中的物体交互。
  - 单视角学习可供性：**AffordanceLLM**, [paper](https://arxiv.org/pdf/2401.06341)：通过利用预训练的大规模视觉语言模型中的丰富知识，显著提升了物体可供性锚定在未见对象和动作上的泛化能力。
  - 数据集：**AGD20K**, [page](https://github.com/lhc1224/Cross-View-AG)
- **3D**
  - 基于点云的可供性锚定：**OpenAD**, [paper](https://arxiv.org/pdf/2203.09905)
  - 室内环境任务中的可供性锚定：**SceneFun3D**, [paper](https://arxiv.org/pdf/2401.06341)
  - 点云数据集：**3D AffordanceNet**, [page](https://github.com/lhc1224/Cross-View-AG)，专注于物体层面的可供性锚定。
  - 实物数据集：**SceneFun3D**, [page](https://scenefun3d.github.io/)，强调在真实室内环境的应用。

### Multimodal Models

> 多模态旨在统一来自不同模态信息的表征, 在具身智能中由于面对着机器识别的视觉信息与人类自然语言的引导信息等不同模态的信息，多模态技术愈发重要。

- 最经典的工作CLIP: [知乎](https://zhuanlan.zhihu.com/p/493489688)
- 多模态大语言模型的经典工作 LLaVA: [website](https://llava-vl.github.io/)
- 多模态生成模型综述: [pdf](https://arxiv.org/pdf/2503.04641)
- 多模态大语言模型强化学习项目：VLM-R1: [repo](https://github.com/om-ai-lab/VLM-R1) 来自OmAI Lab的多模态大语言模型DeepSeek R1-style强化学习开源项目，使用GRPO强化学习算法对多模态大语言模型进行优化，效果优于常规sft，是训练具身智能模型的一种新方向。



### Robot Navigation

**机器人巡航（Robot Navigation）\**是一类要求智能体在\**未知**场景中，通过获取并处理环境信息，实现达成某种目标的路径规划。机器人巡航是具身任务中的一个重要能力，是完成复杂任务不可缺少的基础技术。机器人巡航任务中，智能体一般接受传感器提供的RGB、深度、GPS等信息和相关目标指令，输出是一系列的动作指令。

按照任务类型分类，机器人巡航可以分为以下几个部分：

- **物体目标巡航（Object-Goal Navigation）**：最常见和最广泛的巡航任务。智能体接受的指令是对一个特定物体的描述，目标是找寻到这个物体。
- **图像目标巡航（Image-Goal Navigation）**：智能体接受的指令是一个图像，目标是找寻到这个图像所描述的场景。
- **视觉-语言巡航（Vision-Language Navigation，VLN）**：智能体接受的指令是一个自然语言指令描述，目标是跟随该指令行进。

按照模型架构分类，机器人巡航可以分为以下几个类别：

- **端到端模型（End-to-End Model）**：模型直接将传感器输入通过强化学习或模仿学习映射到动作指令。模型会先将传感器信息编码为视觉表征，结合历史动作作为输入，最后通过与环境交互获得reward实现动作决策的学习。端到端模型主要针对两方面进行优化：一是提升视觉表征能力，二是解决稀疏奖励等动作决策方面的问题。端到端模型的优势在于直截了当，但是面临着严重的过拟合和低泛化性问题，使得其在现实生活中的应用收到了挑战。
  - 经典工作：
    - [Learning Object Relation Graph and Tentative Policy for Visual Navigation](https://arxiv.org/abs/2007.11018)
    - [VTNet: Visual Transformer Network for Object Goal Navigation](https://openreview.net/forum?id=DILxQP08O3B)
- **模块化模型（Modular Model）**：将传感器信息输入不同的模块，模块之间通过接口交互，输出动作指令。模块包括建图模块（Mapping，构建语义和占有地图），长期决策模块（Global Policy，决定长期的导航目标），短期决策模块（Local Policy，决定实现长期目标的具体操作）等。建图模块是模型的核心，包含有网格地图、包含预测的网格地图、图表示地图等多种形式。模块化模型的优势在于模块之间的解耦，大大加强了模型的可解释性。同时，独立的建图模块也使得模型更容易泛化到未知环境。但是模块化模型的建图模块仍然充斥着手动设计的规则，这一定程度上也限制了模型的通用性。
  - 经典工作：
    - [Object Goal Navigation using Goal-Oriented Semantic Exploration](https://arxiv.org/abs/2007.00643) ： SemExp，最早提出语义地图的概念，学习区域和物体之间关联的语义先验，使智能体能够更好地判断目标物体可能在的方向。
    - [PONI: Potential Functions for ObjectGoal Navigation with Interaction-free Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Ramakrishnan_PONI_Potential_Functions_for_ObjectGoal_Navigation_With_Interaction-Free_Learning_CVPR_2022_paper.pdf)：PONI，提出了基于potential functions的语义地图预测，即基于已有的语义地图学习”补全”的完整地图，想象物体最可能在整个房间的哪个位置，使智能体能够迁移在其他样本中观察到的知识。
    - [3D-Aware Object Goal Navigation via Simultaneous Exploration and Identification](https://arxiv.org/abs/2212.00338)：把3D信息编码进巡航的经典工作，通过更精细的点云分割信息，避免了2D语义图在z轴上的信息损失，实现了更精确的语义地图构建。
- **零样本模型（Zero-shot Model）**：模型不接触训练数据，直接在测试阶段完成任务。零样本模型往往利用具有知识先验的大规模预训练模型（CLIP, LLM等）实现。零样本模型的提出旨在解决基于学习的方法面临的过拟合和低泛化性问题，同时也更适合迁移到现实场景。但是零样本模型的缺陷在于推理速度较慢，且性能受限，需要进一步微调以实现更好的性能。
  - 经典工作：
    - [CoWs on Pasture: Baselines and Benchmarks for Language-Driven Zero-Shot Object Navigation](https://arxiv.org/abs/2203.10421)：开放语义物体巡航的提出工作。思路很简单：用CLIP寻找目标物体，找到了就走过去。在不常见物体、复杂描述上取得了很好的效果，同时也有着对不同属性的同类别物体的区分能力。
    - [L3MVN: Leveraging Large Language Models for Visual Target Navigation](https://arxiv.org/abs/2304.05501)：利用LLM决定”我要向哪个边界前进”。利用LLM的人类知识先验，判断物体可能在的房间，以及与其他物体之间的相关关系，实现更快速更有效的巡航。
    - [ESC: Exploration with Soft Commonsense Constraints for Zero-shot Object Navigation](https://arxiv.org/abs/2301.13166)：显式提出了区域对于巡航的影响，在语义地图上标注出区域占有的位置，作为输入的一部分输入给LLM。结合了语义地图连续性和LLM知识丰富的优势。
    - [SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation](https://arxiv.org/abs/2410.08189)：在线构建多层场景图(Scene Graph)并输入给LLM，利用CoT实现LLM对于物体位置的推理。

常用数据集：

- [MatterPort3D(MP3D)](https://niessner.github.io/Matterport/)：真实场景采集，场景复杂庞大，数据量大，难度高。
- [Habitat-Matterport3D(HM3D)](https://aihabitat.org/datasets/hm3d/)：同上
- [RoboTHOR](https://ai2thor.allenai.org/robothor/)：仿真环境，场景小简单。

其他参考：

- [物体目标巡航综述](https://orca.cardiff.ac.uk/id/eprint/167432/1/ObjectGoalNavigationSurveyTASE.pdf)
- [awesome vision-language navigation](https://github.com/eric-ai-lab/awesome-vision-language-navigation)
- [Habitat Navigation Challenge](https://github.com/facebookresearch/habitat-challenge)(Habitat框架中整合了非常多常见的agent skill，例如语义地图构建，FBE和一些heuristic方法，非常适合模块化方法的开发)

## Software

### Simulators

常见仿真器wiki: [wiki](https://simulately.wiki/)

| 仿真器                                             | 对应基准集                                                   |
| :------------------------------------------------- | :----------------------------------------------------------- |
| [IsaacGym](https://developer.nvidia.com/isaac-gym) | [legged gym](https://github.com/leggedrobotics/legged_gym)[parkour(包括蒸馏以及真机部署)](https://github.com/ZiwenZhuang/parkour)[extreme-parkour](https://github.com/chengxuxin/extreme-parkour) |
| [IsaacSim](https://developer.nvidia.com/isaac/sim) | [BEHAVIOR-1K(可跨平台)](https://behavior.stanford.edu/behavior-1k)+[omniGibson(工具链)](https://behavior.stanford.edu/omnigibson/)[ARNOID](https://arnold-benchmark.github.io/) |
| [MuJoCo](https://mujoco.org/)                      | [robosuite](https://robosuite.ai/docs/overview.html)+[robomimic(工具链)](https://robomimic.github.io/)[LIBERO](https://libero-project.github.io/main.html)[MetaWorld](https://meta-world.github.io/)[Gymnasium-Robotics(Fetch; Shadow Dexterous Hand; Maze; Adroit Hand; Franka Kitchen; MaMuJoCo)](https://robotics.farama.org/)[RoboCasa](https://github.com/robocasa/robocasa?tab=readme-ov-file)[RoboHive](https://github.com/vikashplus/robohive) |
| [Sapien](https://sapien.ucsd.edu/)                 | [ManiSkill](https://maniskill.readthedocs.io/en/latest/index.html)[RoboTwin](https://github.com/TianxingChen/RoboTwin) |
| [CoppeliaSim](https://www.coppeliarobotics.com/)   | [RLBench](https://github.com/stepjam/RLBench)[PerAct2](https://bimanual.github.io/)[COLOSSEUM](https://robot-colosseum.github.io/) |
| [PyBullet](https://pybullet.org/wordpress/)        | [Calvin](https://github.com/mees/calvin?tab=readme-ov-file)[Ravens](https://github.com/google-research/ravens)[VimaBench](https://github.com/vimalabs/VimaBench) |
| [Genesis](https://genesis-embodied-ai.github.io/)  |                                                              |
| [SOFA](https://github.com/sofa-framework/sofa/)    | 常用于软体机器人的仿真                                       |

**教程**：

- **Isaac 101：** [Blog](https://axi404.top/tags/isaac 101) by Axi404.



### Banchmarks

具身智能常用benchmark总结 [1]: [zhihu](https://zhuanlan.zhihu.com/p/695342864)

- **CALVIN**, [github](https://github.com/mees/calvin), [website](http://calvin.cs.uni-freiburg.de/)2022年, 第一个公开的结合了自然语言控制、高维多模态输入、7自由度的机械臂控制以及长视野的机器人操纵benchmark。支持不同的语言指令, 不同的摄像头输入, 不同的控制方式, 主要用来评估具身智能模型的多模态输入的能力和长程规划能力。
- **Meta-World**, [webpage](https://meta-world.github.io/): 评估机器人在多任务和元强化学习场景下的表现。50个机器人操作任务(如抓取、推动物体、开门等), 组织成不同的基准测试集(如ML1、ML10、ML45、MT10、MT50等), 每个集合都有明确的训练任务和测试任务。周边和文档比较全面, 基于mojoco, 有完整的API和工具, python import即可运行。
- **Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making**, [website](https://embodied-agent-interface.github.io/): 主要评估大型语言模型(LLMs)在具身决策中的表现, 重点在于决策过程, 包括目标解释、子目标分解、动作序列化和状态转换建模, 不涉及到具体的执行。
- **RoboGen**, [repo](https://github.com/Genesis-Embodied-AI/RoboGen), [website](https://robogen-ai.github.io/): 不是生成policy, 而是生成任务、场景和带标记的数据, 能直接用来监督学习。
- **LIBERO**, [repo](https://github.com/Lifelong-Robot-Learning/LIBERO), [website](https://libero-project.github.io/intro.html): 用一个程序化生成管道来生成任务, 这个管道理论上可以生成无限数量的操作任务, 还提供了：三种视觉运动策略网络架构(RNN、Transformer和ViLT) 和 三种终身学习算法, 以及顺序微调和多任务学习的基准。
- **RoboTwin**, [repo](https://github.com/TianxingChen/RoboTwin): 使用程序生成双臂机器人无限操作任务数据, 并提供了所有任务的评测基准。



### Datasets

- **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**, [website](https://robotics-transformer-x.github.io/): 22种不同机器人平台的超过100万条真实机器人轨迹数据，覆盖了527种不同的技能和160,266项任务，主要集中在抓取和放置。
- **AgiBot World Datasets (智元机器人)**, [website](https://agibot-world.com/): 八十余种日常生活中的多样化技能，超过100万条轨迹数据，采集自**同构型机器人**, 多级质量把控和全程人工在环的策略，从采集员的专业培训，到采集过程中的严格管理，再到数据的筛选、审核和标注，每一个环节都经过了精心设计和严格把控。
- **RoboMIND**, [website](https://x-humanoid-robomind.github.io/): 包含了在479种不同任务中涉及96类独特物体的10.7万条真实世界演示轨迹，来自四种不同协作臂，任务被分为基础技能、精准操作、场景理解、柜体操作和协作任务五大类。
- **All Robots in One,** [website](https://imaei.github.io/project_pages/ario/): ARIO 数据集，包含了 **2D、3D、文本、触觉、声音 5 种模态的感知数据**，涵盖**操作**和**导航**两大类任务，既有**仿真数据**，也有**真实场景数据**，并且包含多种机器人硬件，有很高的丰富度。在数据规模达到三百万的同时，还保证了数据的统一格式，是目前具身智能领域同时达到高质量、多样化和大规模的开源数据集。
- **MimicGen** [26 Oct 2023, CoRL 2023],[repo](https://github.com/NVlabs/mimicgen),[website](https://mimicgen.github.io/)：基于Robosuite与MuJoCo开发的高效数据生成框架，主要聚焦于单臂机器人桌面操作任务，支持多种主流机器人型号。MimicGen提出了一种自动化的数据扩增方法，能够从少量真实人类演示中自动生成大量模拟数据，例如仅使用200段真人演示即可生成超过5万条仿真演示数据，涵盖18类常见机器人任务。
- **RoboCasa** [4 Jun 2024],[repo](https://github.com/robocasa/robocasa), [website](https://robocasa.ai/):基于RoboSuite与MimicGen在MuJoCo中构建的高仿真厨房任务仿真平台。RoboCasa提供了120个多样化厨房环境，包含超过2500个3D物体模型。平台支持单臂、双臂、人形机器人以及移动底座搭载机械臂的机器人系统。此外，RoboCasa内置了25种基础原子任务和75种组合任务，能够真实模拟机器人在复杂厨房场景中的多样化操作行为。
- **DexMimicGen** [6 Mar 2025, ICRA 2025],[repo](https://github.com/NVlabs/dexmimicgen/), [website](https://dexmimicgen.github.io/):以RoboSuite和MimicGen为基础，在MuJoCo平台上构建的高保真双臂桌面操作任务仿真环境。DexMimicGen涵盖9类典型双臂任务，提出了增强版real2sim2real数据自动生成技术，只需60段真实人类演示便可生成2.1万条高质量仿真数据。相比原版MimicGen，该框架显著提升了数据生成效率和真实感，使机器人双臂协作任务的仿真训练更具实用性。

## Paper Lists

- Awesome Humanoid Robot Learning - Yanjie Ze: [repo](https://github.com/YanjieZe/awesome-humanoid-robot-learning)
- Paper Reading List - DeepTimber Community: [repo](https://github.com/DeepTimber-Robot-Lab/Paper-Reading-List)
- Paper List - Yanjie Ze: [repo](https://github.com/YanjieZe/Paper-List)
- Paper List For EmbodiedAI - Tianxing Chen: [repo](https://github.com/TianxingChen/Paper-List-For-EmbodiedAI)
- SOTA Paper Rating - Weiyang Jin: [website](https://waynejin0918.github.io/SOTA-paper-rating.io/)
- Awesome-LLM-Robotics: A repo contains a curative list of papers using Large Language/Multi-Modal Models for Robotics/RL: [website](https://github.com/GT-RIPL/Awesome-LLM-Robotics)



```markdown
[1] Embodied-AI-Guide-Contributors, Lumina-Embodied-AI-Community. Embodied-AI-Guide[EB/OL]. 2025. https://github.com/tianxingchen/Embodied-AI-Guide
```
