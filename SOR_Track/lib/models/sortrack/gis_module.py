# lib/models/ceutrack/gis_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GISModule(nn.Module):
    """
    Class - GIS Module

    将 TMS 对齐后的时间特征 f_temporal 有选择地注入到 SOR 增强后的空间特征 f_spatial 中。
    """

    _CORR_KERNEL = 7 
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # 对互相关后的静态特征响应进行平滑和降维
        self.static_bottleneck = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()    
        )

        # Depth-wise Separable Conv (DSC) 用于生成 Gate
        self.gate_conv = nn.Sequential(
            # Depth-wise
            nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False),
            nn.GELU(),
            # Point-wise
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        
    def _depthwise_correlation(self, f_spatial:torch.Tensor, z_spatial:torch.Tensor) -> torch.Tensor:
        """
        Method - (private)_depthwise_correlation
    
        将 z_spatial 作为核，在 f_spatial 上滑动做相关
        
        Args
        - f_spatial: torch.Tensor, 当前搜索帧特征  [B, C, H, W]
        - z_spatial: torch.Tensor, 当前样板帧特征  [B, C, H_z, W_z]

        Returns
        - corr_map: torch.Tensor, 相关性矩阵

        """
        B, C, H, W = f_spatial.shape
        z_compact = F.adaptive_avg_pool2d(
            z_spatial, (self._CORR_KERNEL, self._CORR_KERNEL)
        )                                                    # [B, C, 7, 7]
        Hz = Wz = self._CORR_KERNEL

        # L2 归一化，使 Correlation 变为 Cosine Similarity
        f_spatial = torch.clamp(f_spatial, -30.0, 30.0)
        z_compact = torch.clamp(z_compact, -30.0, 30.0)
        f_feat = F.normalize(f_spatial, p=2, dim=1, eps=1e-6)
        z_feat = F.normalize(z_compact, p=2, dim=1, eps=1e-6)

        f_reshaped = f_feat.view(1, B * C, H, W)
        z_reshaped = z_feat.view(B * C, 1, Hz, Wz)

        pad_h = (Hz - 1) // 2
        pad_w = (Wz - 1) // 2

        f_padded = F.pad(f_reshaped, 
                         (pad_w, pad_w, pad_h, pad_h))
        corr_map = F.conv2d(
            f_padded,
            z_reshaped,
            padding=0,         
            groups=B * C
        )

        corr_map = corr_map.view(B, C, H, W)
        
        assert corr_map.shape[-2:] == (H, W), \
            f"corr_map shape {corr_map.shape} != expected ({H}, {W})"
        
        return corr_map   
    

    def _dynamic_saliency(self, f_temporal: torch.Tensor) -> torch.Tensor:
        """
        计算 f_temporal 在每个空间位置上的 L2 激活能量，归一化后经 Sigmoid 压缩到 (0, 1)
        Args
            f_temporal : [B, C, H, W]
        Returns
            p_dynamic  : [B, 1, H, W]，值域 (0, 1)
        """
        C = f_temporal.shape[1]
        l2_energy = torch.norm(f_temporal, p=2, dim=1, keepdim=True)   # [B, 1, H, W]
        l2_normed = l2_energy / (C ** 0.5)
        return torch.sigmoid(l2_normed)

    def forward(self, f_temporal: torch.Tensor, f_spatial: torch.Tensor, z_spatial: torch.Tensor) -> torch.Tensor:
        """
        Method - Forward

        Args
        - f_spatial: 当前搜索帧空间特征 [B, C, H, w]
        - f_temporal: 对齐后的时间特征 [B, C, H, W]
        - z_spatial: 模板特征空间锚点 (Template) [B, C, H_z, W_z]
        Returns

        """
        # 获取搜索帧的 Batch Size
        B = f_spatial.shape[0]
        C = f_spatial.shape[1]

        # Batch 对齐
        if z_spatial.shape[0] != B:
            z_spatial = z_spatial.expand(B, -1, -1, -1).contiguous()
        
        if f_temporal.shape[0] != B:
            f_temporal = f_temporal.expand(B, -1, -1, -1).contiguous()
       
        # 基于互相关生成 P_static
        # 获取 [B, C, H, W] 的相关图，每个通道代表该通道下的空间相似度
        corr_map = self._depthwise_correlation(f_spatial, z_spatial)
        # 压缩通道得到 P_static [B, 1, H, W]
        p_static = self.static_bottleneck(corr_map)
        
        # P_dynamic
        p_dynamic = self._dynamic_saliency(f_temporal)  
        
        gate_input = torch.cat([p_static, p_dynamic], dim=1)                 # [B, 2, H, W]
        gate = self.gate_conv(gate_input)                                    # [B, 1, H, W]
        
        out = f_spatial + (gate * f_temporal)

        return out, gate