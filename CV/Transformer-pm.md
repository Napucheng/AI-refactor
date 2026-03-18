# Transformer Update

## nanoGPT

[karpathy/nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs.](https://github.com/karpathy/nanoGPT?spm=a2ty_o01.29997173.0.0.421c5171062mBk)

## llama3

[karpathy/nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs.](https://github.com/karpathy/nanoGPT?spm=a2ty_o01.29997173.0.0.421c5171062mBk)

## flash-attention

偏向CUDA优化

[Dao-AILab/flash-attention: Fast and memory-efficient exact attention](https://github.com/Dao-AILab/flash-attention?spm=a2ty_o01.29997173.0.0.421c5171062mBk)

## VLLM

推理引擎的天花板。学习 **PagedAttention** 如何管理显存。

[vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs](https://github.com/vllm-project/vllm?spm=a2ty_o01.29997173.0.0.421c5171062mBk)



## KV-cache&Quant



## torch.compile?





## Recap

关于transformer的延伸改造和模型：

#### 注意力机制优化

- 稀疏注意力
- 线性注意力
- 滑动窗口
- 分组查询注意力

#### 位置编码

- 固定
- 可学习
- RoPE
- iRoPE

#### 归一化

- postLN
- Pre-LN
- RMSNorm

#### 激活函数

- ReLU
- ReLU
- SwiGLU

#### 架构创新

- MoE

- 并行子层

  ：PaLM, GPT-J

  - Attention 和 FFN 并行计算，提升硬件利用率 ~10-20%

- 稳定性机制

  （2024+ 新增）：

  - QK-Normalization：控制 attention logit 方差，防止训练崩溃
  - Logit Soft-capping：`logits ← c·tanh(logits/c)` 避免数值溢出

#### 语音与多模态

- **Speech Transformer**：端到端 ASR，替代 RNN-T
- **AudioMAE**：掩码自编码预训练，学习通用音频表示

| **Flamingo / LLaVA** | Image + Text               | 视觉问答、指令跟随 |
| -------------------- | -------------------------- | ------------------ |
| **Perceiver IO**     | 任意模态集合               | 通用接口设计       |
| **RT-2 / VoxPoser**  | Vision + Language + Action | 机器人规划与控制   |