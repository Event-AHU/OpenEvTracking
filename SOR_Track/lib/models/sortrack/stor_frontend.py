# lib/models/ceutrack/stor_frontend.py
"""
STOR Frontend：SOR + TMA + GIS 完整前端。

继承 SORFrontend 的 stem/sor/reduction 结构，
在 search 路径上追加 TMA（时序对齐）和 GIS（门控融合）。

数据流:
  Template:
    t_rgb + t_evt → stem → sor(φ=0) → z_sor
                                    ↓
                         reduction → z_feat  [B, D, hz, wz]  → ViT

  Search:
    s_rgb_t + s_evt_t → stem → sor(φ=0) → x_spatial
    s_rgb_prev + s_evt_prev → stem_evt_only → f_evt_prev
                                                    ↓
                         TMA(f_evt_t, f_evt_prev, z_sor) → f_temporal
                         GIS(f_temporal, x_spatial, z_sor) → x_enhanced
                         reduction → x_feat  [B, D, hx, wx]  → ViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.sortrack.stemnet import SPDStem
from lib.models.sortrack.sor_module import SORModule
from lib.models.sortrack.tms_module import TMSModule
from lib.models.sortrack.gis_module import GISModule


class STORFrontend(nn.Module):
    """
    完整 STOR 前端，用于 arch_mode='stor'。

    参数说明:
        in_channels      : 单模态输入通道数 (RGB=3)
        embed_dim        : SPDStem 输出 & ViT 嵌入维度
        stem_scale       : SPDStem 下采样倍率
        sor_K            : SOR Gabor 方向数
        reduction_stride : 额外降采样步长（控制送入 ViT 的 token 数）
        tma_groups       : TMSModule DCNv4 分组数
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        stem_scale: int = 4,
        sor_K: int = 4,
        reduction_stride: int = 2,
        tma_groups: int = 4,
    ):
        super().__init__()
        self.embed_dim        = embed_dim
        self.stem_scale       = stem_scale
        self.sor_K            = sor_K
        self.reduction_stride = reduction_stride

        self.stem_rgb = SPDStem(
            in_channels=in_channels,
            out_channels=embed_dim,
            scale=stem_scale,
        )
        self.stem_evt = SPDStem(
            in_channels=in_channels,
            out_channels=embed_dim,
            scale=stem_scale,
        )

        # SOR
        self.sor = SORModule(
            in_channels=embed_dim,
            K=sor_K,
        )

        # TMA
        self.tma = TMSModule(
            channels=embed_dim,
            group=tma_groups,
        )

        # GIS
        self.gis = GISModule(channels=embed_dim)

        # Reduction 
        if reduction_stride > 1:
            assert reduction_stride % 2 == 0, \
                f"reduction_stride 必须是 2 的幂次，got {reduction_stride}"
            layers    = []
            remaining = reduction_stride
            while remaining > 1:
                layers += [
                    nn.Conv2d(embed_dim, embed_dim,
                              kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GELU(),
                ]
                remaining //= 2
            layers.append(nn.GroupNorm(num_groups=32, num_channels=embed_dim))
            self.reduction = nn.Sequential(*layers)
        else:
            self.reduction = nn.Identity()

    # 公共工具
    def get_token_grid(self, input_size: int) -> int:
        """
        返回给定输入尺寸经 SPDStem + reduction 后的空间边长。
        例: input=256, stem_scale=4, reduction_stride=2 → 64//2=32
        """
        return (input_size // self.stem_scale) // self.reduction_stride

    def _phi_zeros(self, B: int, device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
        """返回全零 phi，用于 SOR 各向均匀扫描。"""
        return torch.zeros(B, device=device, dtype=dtype)

    # Template 路径
    def encode_template(
        self,
        t_rgb: torch.Tensor,   # [B, 3, Hz, Wz]
        t_evt: torch.Tensor,   # [B, 3, Hz, Wz]
    ):
        """
        输出:
            z_feat    : [B, D, hz, wz]  经 reduction，送入 ViT
            z_spatial : [B, D, Hz', Wz'] 未经 reduction，供 GIS 互相关
                        Hz' = Hz // stem_scale

        注意 z_spatial 保留 reduction 前的分辨率，
        GIS._depthwise_correlation 内部 padding 已处理尺寸差异。
        """
        t_rgb = torch.nan_to_num(t_rgb, nan=0.0, posinf=1.0, neginf=-1.0)
        t_evt = torch.nan_to_num(t_evt, nan=0.0, posinf=1.0, neginf=-1.0)
    
        B      = t_rgb.shape[0]
        device = t_rgb.device
        dtype  = t_rgb.dtype
        phi    = self._phi_zeros(B, device, dtype)

        f_rgb     = self.stem_rgb(t_rgb)              # [B, D, Hz', Wz']
        f_evt     = self.stem_evt(t_evt)              # [B, D, Hz', Wz']
        z_spatial = self.sor(f_rgb, f_evt, phi)   # [B, D, Hz', Wz']
        z_feat    = self.reduction(z_spatial)     # [B, D, hz, wz]

        return z_feat, z_spatial

    # Search 路径
    def encode_search(
        self,
        s_rgb_t:    torch.Tensor,   # [B, 3, Hx, Wx]  当前帧 RGB
        s_evt_t:    torch.Tensor,   # [B, 3, Hx, Wx]  当前帧 Event
        s_rgb_prev: torch.Tensor,   # [B, 3, Hx, Wx]  上一帧 RGB
        s_evt_prev: torch.Tensor,   # [B, 3, Hx, Wx]  上一帧 Event
        z_spatial:  torch.Tensor,   # [B, D, Hz', Wz'] 模板空间特征（GIS 锚点）
        f_spatial_prev:  torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        输出:
            x_feat : [B, D, hx, wx]  经 GIS 增强 + reduction，送入 ViT
        """
        s_rgb_t    = torch.nan_to_num(s_rgb_t,    nan=0.0, posinf=1.0, neginf=-1.0)
        s_evt_t    = torch.nan_to_num(s_evt_t,    nan=0.0, posinf=1.0, neginf=-1.0)
        s_rgb_prev = torch.nan_to_num(s_rgb_prev, nan=0.0, posinf=1.0, neginf=-1.0)
        s_evt_prev = torch.nan_to_num(s_evt_prev, nan=0.0, posinf=1.0, neginf=-1.0)
    
        B      = s_rgb_t.shape[0]
        device = s_rgb_t.device
        dtype  = s_rgb_t.dtype 

        # 当前帧：RGB+Event 联合空间增强 
        f_rgb_t     = self.stem_rgb(s_rgb_t)                   # [B, D, Hx', Wx']
        f_evt_t     = self.stem_evt(s_evt_t)                   # [B, D, Hx', Wx']
        x_spatial   = self.sor(f_rgb_t, f_evt_t, phi_motion=None)     # [B, D, Hx', Wx']

        #上一帧：仅 Event，提供运动信号
        if f_spatial_prev is None:
            f_evt_prev   = self.stem_evt(s_evt_prev)              # [B, D, Hx', Wx']
            tma_value    = f_evt_prev
        else:
            tma_value    = f_spatial_prev    

        # TMA
        f_temporal = self.tma(
            f_evt    = f_evt_t,     
            f_prev   = tma_value,   
            z_spatial= z_spatial,   # 模板锚点
        )                            # f_temporal: [B, D, Hx', Wx']

        # GIS：门控融合 
        x_enhanced, _gate = self.gis(
            f_temporal = f_temporal,   # 时序对齐特征
            f_spatial  = x_spatial,    # SOR 空间增强特征
            z_spatial  = z_spatial,    # 模板锚点（互相关基准）
        )                              # x_enhanced: [B, D, Hx', Wx']

        # Reduction 
        x_feat = self.reduction(x_enhanced)   # [B, D, hx, wx]

        return x_feat, x_spatial

    # Forward 入口
    def forward(
        self,
        t_rgb:      torch.Tensor,
        t_evt:      torch.Tensor,
        s_rgb_t:    torch.Tensor,
        s_evt_t:    torch.Tensor,
        s_rgb_prev: torch.Tensor,
        s_evt_prev: torch.Tensor,
        f_spatial_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_feat : [B, D, hz, wz]
            x_feat : [B, D, hx, wx]
        """
        z_feat, z_spatial = self.encode_template(t_rgb, t_evt)
        x_feat, x_spatial = self.encode_search(
            s_rgb_t, s_evt_t,
            s_rgb_prev, s_evt_prev,
            z_spatial,
            f_spatial_prev=f_spatial_prev,     
        )
        return z_feat, x_feat, x_spatial