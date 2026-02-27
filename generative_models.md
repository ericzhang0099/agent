# 生成模型研究 - 扩散模型与VAE

## 目录
1. [扩散模型原理](#1-扩散模型原理)
2. [VAE变分自编码器](#2-vae变分自编码器)
3. [流模型Normalizing Flows](#3-流模型normalizing-flows)
4. [文本到图像生成](#4-文本到图像生成)
5. [代码实现](#5-代码实现)

---

## 1. 扩散模型原理

### 1.1 核心思想

扩散模型（Diffusion Models）通过模拟**前向加噪过程**和**反向去噪过程**来生成数据。

```
前向过程（扩散）:  x₀ → x₁ → x₂ → ... → x_T  (逐渐添加噪声)
反向过程（去噪）:  x_T → x_{T-1} → ... → x₀  (逐步恢复数据)
```

### 1.2 DDPM (Denoising Diffusion Probabilistic Models)

#### 前向过程（马尔可夫链）

前向过程逐步向数据添加高斯噪声，是一个固定的马尔可夫链：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

其中 $\beta_t$ 是噪声调度参数（通常从小到大）。

**关键性质**：可以直接从 $x_0$ 采样任意时刻的 $x_t$：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$

#### 反向过程（学习去噪）

反向过程通过学习神经网络来恢复数据：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

#### 训练目标

DDPM的训练目标是预测噪声（简化版）：

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]$$

其中 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$

### 1.3 DDIM (Denoising Diffusion Implicit Models)

DDIM是DDPM的加速版本，核心改进：

1. **非马尔可夫反向过程**：允许跳步采样
2. **确定性采样**：可以用更少的步数（如50步）生成高质量图像
3. **一致性**：相同的初始噪声产生相似的结果

DDIM的采样公式：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon_t$$

其中 $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$

当 $\sigma_t = 0$ 时，采样变为确定性。

### 1.4 扩散模型架构

```
┌─────────────────────────────────────────────────────────┐
│                    U-Net 架构                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Input x_t  ──┐                                        │
│                │                                        │
│   Time embedding ──┐                                    │
│                    ▼                                    │
│   ┌─────────────────────────────────────┐              │
│   │  Encoder (下采样)                    │              │
│   │  Conv → TimeEmbed → ResBlock → Down │              │
│   │         ↓ Self-Attention            │              │
│   └─────────────────────────────────────┘              │
│                    │                                    │
│                    ▼                                    │
│   ┌─────────────────────────────────────┐              │
│   │  Bottleneck (中间层)                 │              │
│   │  ResBlock → Attention → ResBlock    │              │
│   └─────────────────────────────────────┘              │
│                    │                                    │
│                    ▼                                    │
│   ┌─────────────────────────────────────┐              │
│   │  Decoder (上采样)                    │              │
│   │  Up → ResBlock → TimeEmbed → Conv   │              │
│   │     ↑ Skip Connection               │              │
│   └─────────────────────────────────────┘              │
│                    │                                    │
│                    ▼                                    │
│              Output: 预测噪声 ε_θ                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. VAE变分自编码器

### 2.1 核心思想

VAE（Variational Autoencoder）是一种**生成模型**，结合了深度学习和变分推断。

```
传统自编码器:  x → Encoder → z → Decoder → x̂  (确定性)
VAE:           x → Encoder → μ,σ → z ~ N(μ,σ) → Decoder → x̂  (概率性)
```

### 2.2 数学原理

#### 变分下界（ELBO）

VAE的目标是最大化数据的对数似然的下界：

$$\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$

**ELBO = 重构损失 - KL散度**

#### 重参数化技巧

为了通过梯度下降训练，使用重参数化：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这样可以将随机性从网络参数中分离，使得反向传播可行。

### 2.3 网络架构

```
┌─────────────────────────────────────────────────────────┐
│                      VAE 架构                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   输入 x                                                │
│     │                                                   │
│     ▼                                                   │
│   ┌─────────────────────────────────────┐              │
│   │         Encoder (编码器)             │              │
│   │  Conv → Conv → Flatten → FC         │              │
│   └─────────────────────────────────────┘              │
│     │                                                   │
│     ├──→ μ (均值向量)                                   │
│     │                                                   │
│     └──→ log(σ²) (对数方差)                             │
│           │                                             │
│           ▼                                             │
│     z = μ + σ ⊙ ε  (重参数化采样)                        │
│           │                                             │
│           ▼                                             │
│   ┌─────────────────────────────────────┐              │
│   │         Decoder (解码器)             │              │
│   │  FC → Reshape → Deconv → Deconv     │              │
│   └─────────────────────────────────────┘              │
│           │                                             │
│           ▼                                             │
│   输出 x̂ (重构数据)                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.4 损失函数

```python
# VAE损失 = 重构损失 + KL散度
loss = reconstruction_loss + beta * kl_divergence

# 重构损失 (MSE或BCE)
reconstruction_loss = ||x - x̂||²

# KL散度
kl_divergence = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

### 2.5 VAE vs 扩散模型

| 特性 | VAE | 扩散模型 |
|------|-----|----------|
| 训练速度 | 快 | 慢 |
| 采样速度 | 快（单次前向） | 慢（多步迭代） |
| 生成质量 | 中等 | 高 |
| 隐空间 | 有意义、可插值 | 纯噪声 |
| 模式覆盖 | 可能有缺失 | 完整覆盖 |

---

## 3. 流模型Normalizing Flows

### 3.1 核心思想

流模型通过**可逆变换**将简单分布（如高斯）映射到复杂分布（如图像）。

```
简单分布 z ~ N(0,I)  ──→  可逆变换 f_θ  ──→  复杂分布 x = f_θ(z)
                                         
采样: z ~ N(0,I) → x = f_θ(z)
密度: p_x(x) = p_z(f_θ⁻¹(x)) · |det(J_f⁻¹)|
```

### 3.2 数学原理

#### 变量变换公式

对于可逆变换 $x = f(z)$，密度变换为：

$$p_x(x) = p_z(z) \left| \det \frac{\partial f^{-1}}{\partial x} \right| = p_z(z) \left| \det \frac{\partial f}{\partial z} \right|^{-1}$$

#### 训练目标

最大化对数似然：

$$\mathcal{L} = \mathbb{E}_{x \sim p_{data}} \left[ \log p_z(f^{-1}(x)) + \log \left| \det J_{f^{-1}}(x) \right| \right]$$

### 3.3 可逆变换设计

为了使计算可行，流模型使用特殊结构确保雅可比行列式容易计算：

```
┌─────────────────────────────────────────────────────────┐
│              耦合层 (Coupling Layer)                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输入: [x₁, x₂]  (将输入分成两部分)                      │
│     │                                                   │
│     ├──→ x₁ (保持不变)                                  │
│     │      │                                            │
│     │      └──→ NeuralNet(x₁) → (s, t)                  │
│     │                                                   │
│     └──→ x₂' = x₂ ⊙ exp(s) + t  (仿射变换)              │
│                                                         │
│  输出: [x₁, x₂']                                        │
│                                                         │
│  逆变换:                                                │
│    x₂ = (x₂' - t) ⊙ exp(-s)                             │
│                                                         │
│  雅可比行列式: diag(exp(s), 1) → 容易计算                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.4 常见流模型

| 模型 | 特点 |
|------|------|
| **RealNVP** | 使用耦合层，简单高效 |
| **Glow** | 引入1×1可逆卷积，生成高质量图像 |
| **FFJORD** | 连续时间流，使用神经网络ODE |
| **Flow++** | 改进的耦合层，更好的密度估计 |

### 3.5 流模型 vs 其他生成模型

```
┌────────────────────────────────────────────────────────────┐
│                    生成模型对比                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  GAN:    z → [Generator] → x        (隐式密度，对抗训练)    │
│                                                            │
│  VAE:    x → [Encoder] → z → [Decoder] → x  (近似推断)      │
│                                                            │
│  Flow:   z ↔ [可逆变换] ↔ x          (精确似然，可逆)        │
│                                                            │
│  Diffusion: x₀ → ... → x_T → ... → x₀  (多步去噪)          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 4. 文本到图像生成

### 4.1 CLIP: 连接文本与图像

CLIP（Contrastive Language-Image Pre-training）通过对比学习将文本和图像映射到同一嵌入空间。

```
┌─────────────────────────────────────────────────────────┐
│                    CLIP 架构                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  图像 x ──→ Image Encoder ──→ 图像特征 v_i              │
│                                                         │
│  文本 y ──→ Text Encoder ──→ 文本特征 v_t               │
│                                                         │
│  相似度: s(i,t) = v_i · v_t / (||v_i|| ||v_t||)         │
│                                                         │
│  训练目标: 最大化匹配对的相似度，最小化非匹配对           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.2 文本条件扩散模型

将文本条件引入扩散模型的两种方式：

#### 方式1: Classifier Guidance

使用预训练分类器引导生成：

$$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t) - \sqrt{1-\bar{\alpha}_t} \cdot w \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

#### 方式2: Classifier-Free Guidance (CFG)

更常用的方法，训练时随机丢弃条件：

$$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))$$

其中 $w$ 是引导强度（通常 7.5），$\emptyset$ 表示无条件。

### 4.3 Stable Diffusion 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stable Diffusion 架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  文本输入 ──→ CLIP Text Encoder ──→ 文本特征 (77×768)            │
│                                                                 │
│  图像输入 ──→ VAE Encoder ──→ 潜空间表示 (64×64×4)               │
│                                                                 │
│                    ↓                                            │
│           U-Net with Cross-Attention                            │
│           (噪声预测 + 文本条件)                                   │
│                    ↓                                            │
│           VAE Decoder ──→ 生成图像 (512×512×3)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 生成流程

```python
# 文本到图像生成流程
1. 文本编码: text → CLIP → text_embeddings
2. 初始化噪声: latent = randn(64, 64, 4)
3. 去噪循环 (50步):
   - 预测噪声: noise_pred = UNet(latent, t, text_embeddings)
   - CFG: noise_pred = uncond + w * (cond - uncond)
   - 计算上一步: latent = denoise_step(latent, noise_pred, t)
4. 解码: image = VAE_Decoder(latent)
```

---

## 5. 代码实现

详见 `diffusion_model.py` 文件，包含以下实现：

1. **DDPM 前向/反向过程**
2. **VAE 编码器/解码器**
3. **简化版U-Net噪声预测网络**
4. **图像生成示例**

---

## 参考资料

1. DDPM: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
2. DDIM: "Denoising Diffusion Implicit Models" (Song et al., 2020)
3. VAE: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
4. Normalizing Flows: "Density Estimation using Real NVP" (Dinh et al., 2016)
5. Stable Diffusion: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
6. CLIP: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
