"""
生成模型实现 - 扩散模型与VAE
包含: DDPM, DDIM, VAE, 简化版U-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


# =============================================================================
# 1. 扩散模型 (Diffusion Models) - DDPM
# =============================================================================

class DiffusionScheduler:
    """
    扩散模型调度器 - 管理噪声调度参数
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 预计算常用值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 后验方差
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        
        # 后验均值系数
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向过程: 向干净图像添加噪声
        q(x_t | x_0) = N(x_t; sqrt(alpha_t)*x_0, (1-alpha_t)*I)
        
        Args:
            x0: 干净图像 [B, C, H, W]
            t: 时间步 [B]
            noise: 可选的噪声，如果不提供则随机采样
        
        Returns:
            x_t: 加噪后的图像
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # 获取对应时间步的参数
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
    
    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        从预测的噪声恢复x0
        x_0 = (x_t - sqrt(1-alpha_t)*noise) / sqrt(alpha_t)
        """
        sqrt_recip_alpha = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1_alpha = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_recip_alpha * x_t - sqrt_recipm1_alpha * noise
    
    def q_posterior_mean_variance(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        """
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        
        posterior_mean = coef1 * x0 + coef2 * x_t
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        posterior_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance


class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置编码 - 用于时间步嵌入
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    带时间嵌入的残差块
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    自注意力块
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑为多头注意力格式
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        
        # 计算注意力
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        h = attn @ v
        
        # 重塑回原始形状
        h = h.transpose(2, 3).contiguous().view(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class UNet(nn.Module):
    """
    简化版U-Net用于噪声预测
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (2, 4),
        dropout: float = 0.1,
        time_emb_dim: int = 256
    ):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        
        # 输入投影
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 编码器
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResidualBlock(now_channels, out_ch, time_emb_dim * 4, dropout))
                now_channels = out_ch
                channels.append(now_channels)
                
                # 在指定分辨率添加注意力
                if i in attention_resolutions:
                    self.encoder_blocks.append(AttentionBlock(now_channels))
            
            if i != len(channel_mult) - 1:
                self.downsample_blocks.append(nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1))
                channels.append(now_channels)
        
        # 中间层
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim * 4, dropout),
            AttentionBlock(now_channels),
            ResidualBlock(now_channels, now_channels, time_emb_dim * 4, dropout)
        ])
        
        # 解码器
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = base_channels * mult
            
            for j in range(num_res_blocks + 1):
                self.decoder_blocks.append(ResidualBlock(now_channels + channels.pop(), out_ch, time_emb_dim * 4, dropout))
                now_channels = out_ch
                
                if i in attention_resolutions:
                    self.decoder_blocks.append(AttentionBlock(now_channels))
            
            if i != len(channel_mult) - 1:
                self.upsample_blocks.append(nn.ConvTranspose2d(now_channels, now_channels, 4, stride=2, padding=1))
        
        # 输出
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # 时间嵌入
        time_emb = self.time_embed(timesteps)
        
        # 输入
        h = self.input_conv(x)
        
        # 编码器
        hs = [h]
        for module in self.encoder_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, time_emb)
            else:
                h = module(h)
            hs.append(h)
        
        for module in self.downsample_blocks:
            h = module(h)
            hs.append(h)
        
        # 中间层
        for module in self.middle_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, time_emb)
            else:
                h = module(h)
        
        # 解码器
        upsample_idx = 0
        for module in self.decoder_blocks:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, time_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            elif isinstance(module, nn.ConvTranspose2d):
                h = module(h)
                upsample_idx += 1
        
        return self.output_conv(h)


class DDPM:
    """
    DDPM模型 - 去噪扩散概率模型
    """
    def __init__(self, model: UNet, scheduler: DiffusionScheduler, device: str = 'cuda'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
        self.num_timesteps = scheduler.num_timesteps
    
    def training_step(self, x0: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        单步训练
        """
        self.model.train()
        optimizer.zero_grad()
        
        B = x0.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        
        # 采样噪声
        noise = torch.randn_like(x0)
        
        # 前向加噪
        x_t = self.scheduler.add_noise(x0, t, noise)
        
        # 预测噪声
        noise_pred = self.model(x_t, t)
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        DDPM采样（反向过程）
        """
        self.model.eval()
        
        # 从纯噪声开始
        x = torch.randn(shape, device=self.device)
        
        # 确定采样步数
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
            timesteps = list(range(self.num_timesteps))[::-1]
        else:
            # 均匀采样步数
            step_ratio = self.num_timesteps // num_inference_steps
            timesteps = list(range(0, self.num_timesteps, step_ratio))[::-1][:num_inference_steps]
        
        for t_idx in timesteps:
            t = torch.full((shape[0],), t_idx, device=self.device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.model(x, t)
            
            # 预测x0
            pred_x0 = self.scheduler.predict_x0_from_noise(x, t, noise_pred)
            
            # 计算后验均值和方差
            model_mean, model_variance, model_log_variance = self.scheduler.q_posterior_mean_variance(pred_x0, x, t)
            
            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # 采样x_{t-1}
            x = model_mean + torch.exp(0.5 * model_log_variance) * noise
        
        return x


class DDIM:
    """
    DDIM采样器 - 加速扩散模型采样
    """
    def __init__(self, model: UNet, scheduler: DiffusionScheduler, device: str = 'cuda'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
        self.num_timesteps = scheduler.num_timesteps
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM采样
        
        Args:
            shape: 输出形状
            num_inference_steps: 采样步数（远小于训练步数）
            eta: 随机性参数，0为确定性，1为标准DDPM
        """
        self.model.eval()
        
        # 从纯噪声开始
        x = torch.randn(shape, device=self.device)
        
        # 均匀采样时间步
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_ratio))[::-1][:num_inference_steps]
        
        for i, t_idx in enumerate(timesteps):
            t = torch.full((shape[0],), t_idx, device=self.device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.model(x, t)
            
            # 预测x0
            alpha_t = self.scheduler.alphas_cumprod[t_idx]
            alpha_prev = self.scheduler.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # 计算方向
            dir_xt = torch.sqrt(1 - alpha_prev - eta**2 * (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) * noise_pred
            
            # 随机噪声
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            else:
                noise = 0
                sigma_t = 0
            
            # DDIM更新公式
            x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        return x


# =============================================================================
# 2. VAE 变分自编码器
# =============================================================================

class VAEEncoder(nn.Module):
    """
    VAE编码器 - 将图像编码为潜在分布
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (32, 64, 128, 256)
    ):
        super().__init__()
        
        modules = []
        now_channels = in_channels
        
        # 下采样卷积层
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(now_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            now_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # 计算编码后的特征维度
        # 假设输入为64x64，经过4次下采样后为4x4
        self.feature_dim = hidden_dims[-1] * 4 * 4
        
        # 均值和对数方差
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回均值和对数方差
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE解码器 - 从潜在变量重构图像
    """
    def __init__(
        self,
        latent_dim: int = 512,
        out_channels: int = 3,
        hidden_dims: Tuple[int, ...] = (256, 128, 64, 32)
    ):
        super().__init__()
        
        # 从潜在变量恢复特征图
        self.feature_dim = hidden_dims[0] * 4 * 4
        self.fc = nn.Linear(latent_dim, self.feature_dim)
        
        # 上采样反卷积层
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        # 最后一层
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1),
                nn.Sigmoid()  # 输出范围[0,1]
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在变量重构图像
        """
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)  # 假设编码器输出256通道
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    """
    变分自编码器完整实现
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (32, 64, 128, 256),
        beta: float = 1.0
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta  # KL散度权重
        
        self.encoder = VAEEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = VAEDecoder(latent_dim, in_channels, tuple(reversed(hidden_dims)))
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码并采样潜在变量
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码潜在变量
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            recon: 重构图像
            mu: 均值
            logvar: 对数方差
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def loss_function(self, recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        VAE损失 = 重构损失 + beta * KL散度
        """
        # 重构损失 (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        
        return recon_loss + self.beta * kl_loss
    
    def sample(self, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        """
        从先验分布采样生成图像
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        在两个图像之间进行插值
        """
        mu1, _ = self.encoder(x1)
        mu2, _ = self.encoder(x2)
        
        interpolations = []
        for alpha in torch.linspace(0, 1, num_steps):
            z = alpha * mu1 + (1 - alpha) * mu2
            recon = self.decode(z)
            interpolations.append(recon)
        
        return torch.stack(interpolations, dim=0)


# =============================================================================
# 3. 使用示例
# =============================================================================

def demo_ddpm():
    """
    DDPM使用示例
    """
    print("=" * 60)
    print("DDPM Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=2
    )
    
    scheduler = DiffusionScheduler(num_timesteps=1000)
    ddpm = DDPM(model, scheduler, device)
    
    # 模拟训练
    print("\n模拟训练...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 生成随机训练数据
    x0 = torch.randn(4, 3, 64, 64, device=device)
    
    for step in range(5):
        loss = ddpm.training_step(x0, optimizer)
        print(f"Step {step + 1}, Loss: {loss:.4f}")
    
    # 采样
    print("\n采样生成图像...")
    samples = ddpm.sample(shape=(4, 3, 64, 64), num_inference_steps=100)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    print("\nDDPM Demo 完成!")


def demo_ddim():
    """
    DDIM使用示例
    """
    print("\n" + "=" * 60)
    print("DDIM Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=2
    )
    
    scheduler = DiffusionScheduler(num_timesteps=1000)
    ddim = DDIM(model, scheduler, device)
    
    # 采样 (使用更少的步数)
    print("\nDDIM采样 (50步)...")
    samples = ddim.sample(shape=(4, 3, 64, 64), num_inference_steps=50, eta=0.0)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    print("\nDDIM Demo 完成!")


def demo_vae():
    """
    VAE使用示例
    """
    print("\n" + "=" * 60)
    print("VAE Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建VAE模型
    vae = VAE(
        in_channels=3,
        latent_dim=512,
        hidden_dims=(32, 64, 128, 256),
        beta=1.0
    ).to(device)
    
    # 模拟训练
    print("\n模拟训练...")
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    # 生成随机训练数据
    x = torch.randn(8, 3, 64, 64, device=device)
    
    vae.train()
    for step in range(5):
        optimizer.zero_grad()
        recon, mu, logvar = vae(x)
        loss = vae.loss_function(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}, Loss: {loss.item():.4f}")
    
    # 从先验采样
    print("\n从先验分布采样...")
    vae.eval()
    with torch.no_grad():
        samples = vae.sample(num_samples=4, device=device)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # 插值
    print("\n图像插值...")
    with torch.no_grad():
        x1 = torch.randn(1, 3, 64, 64, device=device)
        x2 = torch.randn(1, 3, 64, 64, device=device)
        interpolations = vae.interpolate(x1, x2, num_steps=5)
    print(f"Interpolations shape: {interpolations.shape}")
    
    print("\nVAE Demo 完成!")


def demo_comparison():
    """
    模型对比
    """
    print("\n" + "=" * 60)
    print("模型对比")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 统计参数量
    unet = UNet(in_channels=3, out_channels=3, base_channels=64, channel_mult=(1, 2, 4))
    vae = VAE(in_channels=3, latent_dim=512, hidden_dims=(32, 64, 128, 256))
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"\nU-Net (DDPM/DDIM) 参数量: {count_params(unet):,}")
    print(f"VAE 参数量: {count_params(vae):,}")
    
    # 计算效率对比
    print("\n采样效率对比:")
    print("  DDPM: 1000步 (慢，高质量)")
    print("  DDIM: 50步 (快，高质量)")
    print("  VAE: 1步 (最快，中等质量)")
    
    print("\n模型对比完成!")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行演示
    demo_ddpm()
    demo_ddim()
    demo_vae()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)
