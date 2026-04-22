[(40 封私信 / 80 条消息) 生成模型(四):扩散模型 - 知乎](https://zhuanlan.zhihu.com/p/499206074)

------

# 一、最底层：生成模型的统一目标

## 1. 概率建模目标

[
p_\theta(x) \approx p_{\text{data}}(x)
]

核心方式：

| 方法          | 本质             |
| ------------- | ---------------- |
| VAE           | 最大似然（ELBO） |
| GAN           | 对抗分布匹配     |
| Diffusion     | score matching   |
| Flow Matching | 学概率流         |

👉 **统一视角（重要）：**

- diffusion / flow matching 本质都是在建模**概率流（probability flow）**

（✖ 博客未明确统一）

------

# 二、Diffusion 基础（DDPM体系）

## 2. Forward process（加噪）

[
q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
]

闭式表达：

[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
]

✔ 博客已详细讲

------

## 3. Reverse process（去噪）

[
p_\theta(x_{t-1}|x_t)
]

本质：

- 学一个条件高斯分布

✔ 博客已讲

------

## 4. 训练目标（ELBO → 简化）

[
\mathbb{E}||\epsilon - \epsilon_\theta(x_t,t)||^2
]

👉 关键 insight：

- diffusion ≈ 预测噪声

✔ 博客已讲

------

# 三、Score Matching 视角（理论核心）

## 5. score function

[
s(x,t) = \nabla_x \log p_t(x)
]

## 6. diffusion 等价于 score learning

[
\epsilon_\theta \leftrightarrow \nabla \log p
]

✔ 博客提到（但不深入）

------

## 7. Denoising Score Matching（DSM）

[
\mathbb{E}||s_\theta(x_t,t) - \nabla \log p_t(x_t)||^2
]

👉 这是 diffusion 的真正理论基础

（✖ 博客没展开推导）

------

# 四、连续时间视角（SDE）【关键缺失】

## 8. SDE 表达

dx = f(x,t)dt + g(t)dW_t

👉 diffusion 是连续随机过程

（✖ 博客未系统讲）

------

## 9. 反向 SDE

[
dx = [f(x,t) - g^2 \nabla \log p_t(x)]dt + g d\bar{W}
]

👉 关键：

- score 控制逆过程

（✖ 博客未讲）

------

## 10. 两种 SDE

- VP-SDE（DDPM）
- VE-SDE（NCSN）

（✖ 博客未系统区分）

------

# 五、ODE 视角（现代核心）

## 11. Probability Flow ODE

\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)

👉 diffusion = ODE（确定性）

（✖ 博客未讲）

------

## 12. 重要意义

- 可以 deterministic sampling
- 可以用数值方法加速

------

# 六、参数化方式（训练稳定性核心）

## 13. ε-pred（原始）

✔ 博客提到

------

## 14. x₀-pred

✔ 博客提到

------

## 15. v-pred（现代主流）

[
v = \alpha \epsilon - \sigma x_0
]

👉 优势：

- 更稳定
- 大模型必用（Stable Diffusion v2）

（✖ 博客未提）

------

# 七、采样方法（性能关键）

## 16. DDPM（随机）

✔ 博客提到

------

## 17. DDIM（确定性）

✔ 博客提到

------

## 18. 高级 ODE 求解器

- DPM-Solver
- Heun method
- Runge-Kutta
- EDM sampler

👉 本质：

- 数值解 ODE

（✖ 博客未讲）

------

# 八、Guidance（生成质量关键）

## 19. Classifier guidance

✔ 博客提到

------

## 20. Classifier-Free Guidance（CFG）

[
\hat{\epsilon} = (1+w)\epsilon_{cond} - w\epsilon_{uncond}
]

✔ 博客提到

------

# 九、架构（模型设计）

## 21. U-Net（主流）

✔ 博客提到

------

## 22. Transformer（现代趋势）

- DiT（Diffusion Transformer）

👉 替代 U-Net

（✖ 博客未深入）

------

# 十、Latent Diffusion（工业级关键）

## 23. latent 空间 diffusion

👉 pipeline：

- image → VAE encoder → latent
- latent diffusion
- decoder 还原

✔ 博客提到

------

## 24. 代表模型

- Stable Diffusion

GitHub：

- https://github.com/CompVis/stable-diffusion

------

# 十一、多模态（文本生成图像）

## 25. CLIP conditioning

- text embedding → cross-attention

👉 关键组件：

- CLIP

（✖ 博客未系统讲）

------

# 十二、Flow Matching（你重点）

## 26. 基本形式

\frac{dx}{dt} = v_\theta(x,t)

------

## 27. 训练目标

给定路径：

[
x_t = (1-t)x_0 + tz
]

监督：

[
v_\theta(x_t,t) \approx \frac{dx_t}{dt}
]

------

## 28. 和 diffusion 的关系（核心）

[
v = f - \frac{1}{2}g^2 \nabla \log p
]

👉 结论：

> Flow Matching ≈ Diffusion 的 ODE 版本

（✖ 博客完全没讲）

------

## 29. 优势

- 无噪声（deterministic）
- 更稳定
- 更快

------

## 30. 重要论文

- Flow Matching for Generative Modeling

GitHub：

- https://github.com/facebookresearch/flow_matching

------

## 31. Rectified Flow（重要）

👉 思想：

- 让路径更直
- 更容易学习

GitHub：

- https://github.com/gnobitab/RectifiedFlow

（✖ 博客未提）

------

# 十三、Diffusion 扩展方向

## 32. Video Diffusion

- 时序一致性
- 3D attention

（✖ 博客未涉及）

------

## 33. 3D Diffusion

- NeRF + diffusion
- SDS（Score Distillation Sampling）

👉 代表：

- DreamFusion

------

## 34. 4D生成（动态场景）

- dynamic NeRF
- video → 3D

------

# 十四、训练优化（工程关键）

## 35. noise schedule

- linear
- cosine（更好）

✔ 博客提到

------

## 36. loss weighting

- SNR weighting（关键）

（✖ 博客未讲）

------

## 37. EMA（指数滑动平均）

- 稳定训练

（✖ 博客未讲）

------

## 38. mixed precision / large batch

（✖ 博客未讲）

------

# 十五、加速技术（工业级）

## 39. distillation

- progressive distillation

✔ 博客提到（简单）

------

## 40. consistency model

👉 一步生成

（✖ 博客未讲）

------

## 41. latent consistency model（LCM）

👉 Stable Diffusion 加速

（✖ 博客未讲）

------

# 十六、当前最前沿趋势（2024+）

## 42. Diffusion → Transformer

- DiT

------

## 43. Flow Matching 崛起

👉 可能替代 diffusion

------

## 44. 多模态统一模型

- text + image + video + 3D

------

## 45. Diffusion for World Model

👉 用于：

- 视频预测
- 动态建模

------

# 十七、推荐 GitHub（高质量）

## Diffusion

- https://github.com/openai/guided-diffusion
- https://github.com/huggingface/diffusers
- https://github.com/CompVis/stable-diffusion

------

## Flow Matching

- https://github.com/facebookresearch/flow_matching
- https://github.com/gnobitab/RectifiedFlow

------

## 3D / Video

- https://github.com/ashawkey/stable-dreamfusion
- https://github.com/nerfstudio-project/nerfstudio

------

# 十八、最终总结（你必须记住）

## diffusion 的本质：

> 学 score（概率梯度） + SDE 逆过程

------

## flow matching 的本质：

> 学 velocity（概率流）

------

## 二者统一：

> diffusion = stochastic flow
> flow matching = deterministic flow

------

# 如果你要再往“研究级”走

我建议你下一步直接让我带你做：

👉 推导三件事（非常关键）：

1. DDPM → SDE
2. SDE → ODE
3. diffusion ↔ flow matching 等价

或者我可以给你：

👉 一份“论文阅读路线（从入门到CVPR级）”

直接说你要哪条，我给你继续往下打通。