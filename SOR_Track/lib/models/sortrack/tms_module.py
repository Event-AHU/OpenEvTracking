# lib/models/ceutrack/tms_module.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from DCNv4.modules.dcnv4 import DCNv4
    from DCNv4.functions import DCNv4Function
    print('from DCNv4.modules.dcnv4 import DCNv4 success.')
except ImportError:
    print('from DCNv4.modules.dcnv4 import DCNv4 failed. Trying to import DCNv4 directly.')
    try:
        from DCNv4 import DCNv4
    except ImportError:
        print('import DCNv4 failed. Please install DCNv4.')
        DCNv4 = None
        import sys
        sys.exit(1)

class LightingConv(nn.Module):
    """
    Class - LightingConv

    轻量卷积，用于提纯特征
    """
    def __init__(self, channels):
        super().__init__()                                                                               
        # Depth-wise 卷积变种
        self.rectify = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                    groups=channels, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=channels),
            nn.GELU(),
        )

        self.excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.rectify(x)
        gate = self.excitation(x)
        out = x * (1.0 + gate)
        return 50.0 * torch.tanh(out / 50.0)
    
class TMSModule(nn.Module):
    """
    Class - TMSModule

    利用 DCNv4 将上一帧特征 f_spatial(t-1) 对齐到当前帧 t
    """
    def __init__(self, channels:int, group:int=4):
        """
        Method - __init__

        Args
        - channels: int, 输入特征的通道数
        - group: int, DCNv4 的 group 数
        - offset_scale: float, DCNv4 的 offset_scale 数
        """
        super().__init__()
 
        # DCNv4 要求 channels 必须能被 group 整除，且 channels // group 建议是 16 的倍数
        assert channels % group == 0, f"channels {channels} must be divisible by group {group}"
        self._GN_GROUP_CH = 32  
        self.channels = channels
        self._num_gn_groups = channels // self._GN_GROUP_CH  

        # Offset 网络：接收 F_evt(t) [B, C, H, W] 和 Z_feat [B, C, H, W] 
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, groups=group),
            nn.GroupNorm(num_groups=self._num_gn_groups, num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=group),
            nn.GroupNorm(num_groups=self._num_gn_groups, num_channels=channels),
            nn.GELU(), 
        )

        self.dcn_core = DCNv4(
            channels=channels, 
            group=group, 
            kernel_size=3, 
            padding=1,
            without_pointwise=False
        )
           
        # Lighting Conv
        self.light_conv = LightingConv(channels)
        
    
    def forward(self, f_evt: torch.Tensor, f_prev: torch.Tensor, z_spatial: torch.Tensor) -> torch.Tensor:
        """
        Method - forward

        Args
        - f_evt: 当前帧事件特征 [B, C, H, W]
        - f_prev: 上一帧空间特征 [B, C, H, W]
        - z_feat: 模板全局特征 [B, C, 1, 1]

        Returns
        
        """
        B, C, H, W = f_evt.shape
        L = H * W 
           
        #    清洗非法值，不截断合法梯度
        f_evt  = torch.nan_to_num(f_evt,  nan=0.0, posinf=1.0, neginf=-1.0)
        f_prev = torch.nan_to_num(f_prev, nan=0.0, posinf=1.0, neginf=-1.0)
   
        if z_spatial.dim() == 4:
            z_global = F.adaptive_avg_pool2d(z_spatial, (1, 1))
        elif z_spatial.dim() == 2:
            z_global = z_spatial.view(B, C, 1, 1)
        else:
            z_global = z_spatial

        z_feat = z_global.expand(B, -1, H, W).contiguous()

        # Offset 网络 - 用于 DCN，不参与 phi
        motion_guidance = self.offset_net(
            torch.cat([f_evt, z_feat], dim=1)
        )

        motion_guidance = torch.tanh(motion_guidance) * 4.0
        # DCNv4 对齐 - DCNv4 预期 (N, L, C)
        # (B, C, H, W) -> (B, H, W, C) -> (B, L, C)
        v_seq = f_prev.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        g_seq = motion_guidance.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        
        # DCN 逻辑
        x = self.dcn_core.value_proj(v_seq)
        x = x.reshape(B, H, W, -1)
        dcn_offset_mask = self.dcn_core.offset_mask(g_seq).reshape(B, H, W, -1)

        f_aligned = DCNv4Function.apply(
            x, dcn_offset_mask,
            self.dcn_core.kernel_size, self.dcn_core.kernel_size,
            self.dcn_core.stride, self.dcn_core.stride,
            self.dcn_core.pad, self.dcn_core.pad,
            self.dcn_core.dilation, self.dcn_core.dilation,
            self.dcn_core.group, self.dcn_core.group_channels,
            self.dcn_core.offset_scale,
            256, # im2col_step
            self.dcn_core.remove_center
        )

        f_aligned = f_aligned.view(B, L, -1)
        f_aligned = self.dcn_core.output_proj(f_aligned)

        f_aligned = f_aligned.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # 特征提纯
        f_temporal = self.light_conv(f_aligned)
        
        return f_temporal