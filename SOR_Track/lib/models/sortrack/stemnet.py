# lib/models/stortrack/stemnet.py

import torch
import torch.nn as nn

class SPDStem(nn.Module):
    """
    Class - SPDStem
    实现 Space-to-Depth 采样。
    """
    def __init__(self, in_channels, out_channels, scale=4):
        """
        Args:
        scale: 下采样倍率 e.g.: input 256, scale=4 -> output 64
        """
        super().__init__()
        self.scale = scale
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=scale)
        
        # 线性投影：将折叠后的 4*C 或 16*C 通道映射到模型维度 D
        folded_channels = in_channels * (scale ** 2)
        num_groups = min(scale ** 2, out_channels // 8)
        assert out_channels % num_groups == 0, \
            f"out_channels={out_channels} 不能被 num_groups={num_groups} 整除"
        
        self.proj = nn.Sequential(
            nn.Conv2d(folded_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.unshuffle(x) # [B, C*scale^2, H/scale, W/scale]
        x = self.proj(x)      # [B, D, H/scale, W/scale]
        return x