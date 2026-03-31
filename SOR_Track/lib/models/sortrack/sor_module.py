# lib/models/stortrack/sor_module.py

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GaborKernelGenerator(nn.Module):
    """
    Class - GaborKernelGenerator

    ODA (Orthogonal Directional Atomic) 核心：
    根据输入的 Phi (主运动方向) 生成 K 个正交方向的 Gabor 卷积核
    Gabor 核公式：
        gb = exp(-0.5 * (x_theta^2 + gamma^2 * y_theta^2) / sigma^2) * cos(2*pi*x_theta/lambd + psi)
    """

    def __init__(self, kernel_size:int=7, K:int=4, 
                 sigma:float=3.0, lambd:float=5.0, gamma:float=0.5, psi:float=0.0):
        """
        Method - __init__

        Args
        - kernel_size: int, default=7, 卷积核大小
        - K: int, default=4, 正交方向数
        - sigma: float, default=3.0, 标准差
        - lambd: float, default=5.0, 波长
        - gamma: float, default=0.5, 方差
        - psi: float, default=0.0, 偏转角度
        """
        super(GaborKernelGenerator, self).__init__()
        self.kernel_size = kernel_size
        self.K = K
        # 可学习超参
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.lambd = nn.Parameter(torch.tensor(lambd))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.psi   = nn.Parameter(torch.tensor(psi)) 

    def forward(self, phi_motion:torch.Tensor)->torch.Tensor:
        """
        Method - forward

        Args
        - phi_motion: torch.Tensor, shape=[B] or [B, 1], 输入的主运动方向

        Return
        - gb: torch.Tensor, shape=[B*K, 1, K_s, K_s], 动态卷积核
        """
        if phi_motion.dim() == 0:
            phi_motion = phi_motion.unsqueeze(0)   # [] → [1]
        elif phi_motion.dim() > 1:
            phi_motion = phi_motion.reshape(-1)    # [B,1] → [B]
        device = phi_motion.device
        B = phi_motion.shape[0]                    
        K = self.K
        ks = self.kernel_size

        # 计算 theta 角
        k_indices = torch.arange(K, device=device).view(1, K)       # [1, K]
        thetas = phi_motion.view(B, 1) + k_indices * math.pi / K    # [B, K]
        thetas = thetas.view(-1)                                    # [B*K]   
        
        # 生成坐标网格
        ymax, xmax = ks // 2, ks // 2
        ymin, xmin = -ymax, -xmax
        x, y = torch.meshgrid(torch.arange(xmin, xmax+1, device=device), 
                              torch.arange(ymin, ymax+1, device=device), 
                              indexing='ij')
        x = x.flatten().repeat(B * K, 1)                              # [B*K, K_s^2]
        y = y.flatten().repeat(B * K, 1)                              # [B*K, K_s^2]

        # 旋转坐标
        cos_t = torch.cos(thetas).unsqueeze(1)                        # [B*K, 1]
        sin_t = torch.sin(thetas).unsqueeze(1)                        # [B*K, 1]
        x_theta = x * cos_t + y * sin_t          
        y_theta = -x * sin_t + y * cos_t         

        sigma_safe = self.sigma.clamp(min=0.5, max=10.0)   
        lambd_safe = self.lambd.clamp(min=1.0, max=20.0)  
        gamma_safe = self.gamma.clamp(min=0.1, max=2.0)
        psi_safe   = self.psi.clamp(-math.pi, math.pi)   
        
        # 生成 Gabor 核
        # gb = exp(-0.5 * (x_theta^2 + gamma^2 * y_theta^2) / sigma^2) * cos(2*pi*x_theta/lambd + psi)
        gb = torch.exp(-0.5 * (x_theta**2 + gamma_safe**2 * y_theta**2) / sigma_safe**2) * \
            torch.cos(2 * math.pi * x_theta / lambd_safe + self.psi)
        gb = gb.view(B, K, 1, ks, ks)                                  # [B, K, 1, K_s, K_s]
        return gb
        
class DirectionHead(nn.Module):
    """
    Class - DirectionHead
    轻量方向预测头：从 f_rgb 与 f_evt 的全局特征中预测主运动方向 phi
    """
    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(in_channels // 4, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)   # 空间压缩，不引入参数
        # 两路特征独立投影后拼接
        self.proj_rgb = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.proj_evt = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        # 融合头：输出单标量
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1, bias=True),
        )
        self._zero_init()

    def _zero_init(self):
        """最后一层零初始化"""
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self,
        f_rgb: torch.Tensor,   # [B, C, H, W]
        f_evt: torch.Tensor,   # [B, C, H, W]
    ) -> torch.Tensor:
        """
        Returns
            phi : [B]，主运动方向（弧度），值域 (-π, π)
        """
        # 全局特征提取
        g_rgb = self.pool(f_rgb).flatten(1)        # [B, C]
        g_evt = self.pool(f_evt).flatten(1)        # [B, C]
        h_rgb = self.proj_rgb(g_rgb)               # [B, hidden]
        h_evt = self.proj_evt(g_evt)               # [B, hidden]
        fused = torch.cat([h_rgb, h_evt], dim=1)   # [B, hidden*2]
        raw   = self.fusion(fused).squeeze(1)      # [B]
        
        return math.pi * torch.tanh(raw)
    
class SORModule(nn.Module):
    """
    Class - SORModule

    基于 Gabor 核的 SOR(Spatial Orthogonal Refinement) 模块
    """
    def __init__(self, in_channels:int, K:int=4):
        """
        Method - __init__

        Args
        - in_channels: int, 输入通道数
        - K: int, 正交方向数
        """
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.kernel_gen = GaborKernelGenerator(K=K)
        self._GROUP_CHANNELS = 32

        assert in_channels % self._GROUP_CHANNELS == 0, f"in_channels:{in_channels} should br divided by current _GROUP_CHANNELS:{self._GROUP_CHANNELS}"

        self.direction_head = DirectionHead(in_channels)
        # 组归一化
        num_groups = K * (in_channels // self._GROUP_CHANNELS)
        self.gn_rgb = nn.GroupNorm(num_groups=num_groups,  num_channels=in_channels * K)
        self.gn_event = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels * K)

        # 1 x 1 Conv
        self.proj = nn.Conv2d(in_channels * K, in_channels, kernel_size=1, bias=False)
        self.post_norm = nn.GroupNorm(in_channels // 32, in_channels)

    # def _apply_directional_conv(self, x, kernels):
    #     """
    #     Method - (private)_apply_directional_conv

    #     Args
    #     - x: torch.Tensor, shape=[B, C, H, W], 输入特征图
    #     - kernels: torch.Tensor, shape=[B*K, 1, K_s, K_s], 动态卷积核

    #     Return
    #     - out: torch.Tensor, shape=[B, C * K, H, W], 输出特征图
    #     """
    #     B, C, H, W = x.shape
    #     K = self.K
    #     ks = self.kernel_gen.kernel_size
        
    #     # 准备输入: [B, C, H, W] -> [1, B*C, H, W]
    #     x_reshaped = x.view(1, B * C, H, W)

    #     # 卷积核准备: [B, K, 1, K_s, K_s] -> [B * K * C, K_s, K_s] 
    #     # 每个 Batch 有 K 个核, 我们需要让该 Batch 的 C 个通道都用这 K 个核。
    #     # NOTE: 最初实现，不过貌似 torch 不转置批维度运算会有混淆
    #     # kernels = kernels.repeat_interleave(C, dim=0) # [B*C, K, 1, ks, ks] 
    #     # kernels = kernels.unsqueeze(1)                      # [B, 1, K, 1, K_s, K_s]
    #     # kernels = kernels.expand(B, C, K, 1, ks, ks)          # [B, C, K, 1, K_s, K_s]
    #     # kernels = kernels.reshape(B * C * K, 1, ks, ks)     # [B * C * K, 1, K_s, K_s]
    
    #     kernels = kernels.unsqueeze(2)                      # [B, K, 1, 1, K_s, K_s]    
    #     kernels = kernels.expand(B, K, C, 1, ks, ks)        # [B, K, C, 1, K_s, K_s]
    #     kernels = kernels.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B, C, K, 1, K_s, K_s]
    #     kernels = kernels.view(B * C * K, 1, ks, ks)           # [B * C * K, 1, K_s, K_s]
    
    #     padding = ks // 2
    #     out = F.conv2d(x_reshaped, kernels, padding=padding, groups=B * C)
    #     # out = out.view(B, C * K, H, W)
    #     out = out.view(B, C, K, H, W)
    #     out = out.permute(0, 2, 1, 3, 4).contiguous()
    #     out = out.view(B, K * C, H, W)
    #     return out

    def _apply_directional_conv(
        self,
        x: torch.Tensor,        # [B, C, H, W]
        kernels: torch.Tensor,  # [B, K, 1, ks, ks]
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        K  = self.K
        ks = self.kernel_gen.kernel_size
        pad = ks // 2

        outs = []
        for b in range(B):
            # kernels[b]: [K, 1, ks, ks]
            w_b = kernels[b]                                    # [K, 1, ks, ks]
            w_b = w_b.unsqueeze(1)                              # [K, 1, 1, ks, ks]  
            w_b = w_b.expand(K, C, 1, ks, ks)                  # [K, C, 1, ks, ks]  
            w_b = w_b.permute(1, 0, 2, 3, 4).contiguous()      # [C, K, 1, ks, ks]
            w_b = w_b.view(C * K, 1, ks, ks)                   # [C*K, 1, ks, ks]

            out_b = F.conv2d(
                x[b:b+1],    # [1, C, H, W]
                w_b,         # [C*K, 1, ks, ks]
                padding=pad,
                groups=C,
            )                # [1, C*K, H, W]

            out_b = out_b.view(1, C, K, H, W) \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .view(1, K * C, H, W)
            outs.append(out_b)

        return torch.cat(outs, dim=0)   # [B, K*C, H, W]

    def forward(
            self, 
            f_rgb:torch.Tensor, 
            f_event:torch.Tensor, 
            phi_motion:torch.Tensor | None = None
        ) -> torch.Tensor:
        """
        Method - forward
        
        Args
        - f_rgb: torch.Tensor, shape=[B, C, H, W], 输入的 RGB 特征图
        - f_event: torch.Tensor, shape=[B, C, H, W], 输入的事件特征图
        - phi_motion: torch.Tensor, shape=[B], 输入的主运动方向

        Return

        """
        B, C, H, W = f_rgb.shape
        K = self.K

        # 方向预测
        if phi_motion is None:
            phi = self.direction_head(f_rgb, f_event)   # [B]，梯度完整
        else:
            if phi_motion.dim() > 1:
                phi_motion = phi_motion.reshape(-1)
            phi = phi_motion                             # 外部传入时不走 direction_head
        
        kernels = self.kernel_gen(phi)

        # 卷积多方向响应提取
        
        # # NOTE: For 循环实现版本，时间换空间
        # # HACK: For 循环实现
        # out_rgb_list = []
        # out_event_list = []

        # for b in range(B):
        #     k_b = kernels[b * K : (b + 1) * K]  # [K, 1, K_s, K_s]
        #     # 对当前 batch 的 RGB/EVT 每一个通道都跑 K 个方向
        #     r_b = F.conv2d(f_rgb[b:b+1].view(C, 1, H, W), k_b, padding=self.kernel_gen.kernel_size // 2)  # [C, K, H, W]
        #     e_b = F.conv2d(f_event[b:b+1].view(C, 1, H, W), k_b, padding=self.kernel_gen.kernel_size // 2)  # [C, K, H, W]

        # R_rgb = self.gn_rgb(torch.cat(out_rgb_list, dim=0))
        # R_event = self.gn_event(torch.cat(out_event_list, dim=0))
        # # HACK END

        # NOTE: 并行 Tensor 操作，注意吃显存
        R_rgb = self._apply_directional_conv(f_rgb, kernels)
        R_event = self._apply_directional_conv(f_event, kernels)

        # 组归一化
        R_rgb = self.gn_rgb(R_rgb)     
        R_event = self.gn_event(R_event)

        # if not self.training:
        #     R_event_mean = R_event.mean(dim=[2, 3], keepdim=True)   # [B, C*K, 1, 1]
        #     R_event_std  = R_event.std(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        #     R_event      = (R_event - R_event_mean) / R_event_std   # Z-score
        #     R_event      = R_event * 2.0   # 恢复近似原始幅值（GroupNorm后std≈1，×2→std≈2）
        
        # 门控
        mask = 1.0 + torch.sigmoid(R_event)
        self._last_mask = mask.detach().cpu()
        R_rgb = R_rgb * mask

        f_spatial = self.post_norm(self.proj(R_rgb) + f_rgb)
        self._last_R_rgb_pregn  = R_rgb.detach().cpu() 
        return f_spatial

