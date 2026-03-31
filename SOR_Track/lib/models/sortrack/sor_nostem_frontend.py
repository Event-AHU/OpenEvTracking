# lib/models/ceutrack/sor_nostem_frontend.py
"""
验证 SPDStem 对 SOR 增益的独立贡献
对照组：sor_nostem（无 SPD）vs sor（有 SPD）
"""

import torch
import torch.nn as nn
from lib.models.sortrack.sor_module import SORModule


class SORNoStemFrontend(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        patch_size: int = 16,
        sor_K: int = 4,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.patch_size = patch_size

        # 独立的双路 PatchEmbed
        self.patch_embed_rgb = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim,
                      kernel_size=patch_size, stride=patch_size, bias=False),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
        )
        self.patch_embed_evt = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim,
                      kernel_size=patch_size, stride=patch_size, bias=False),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
        )

        self.sor = SORModule(in_channels=embed_dim, K=sor_K)
        # nostem 无额外 reduction
        self.reduction_stride = 1

    def get_token_grid(self, input_size: int) -> int:
        return input_size // self.patch_size

    def init_from_backbone(self, backbone_patch_embed: nn.Module) -> None:
        """
        用 backbone.patch_embed.proj 的权重热启动双路 embed

        Args:
            backbone_patch_embed: backbone.patch_embed (PatchEmbed 实例)
        """
        with torch.no_grad():
            src_w = backbone_patch_embed.proj.weight   # [D, 3, P, P]
            self.patch_embed_rgb[0].weight.copy_(src_w)
            self.patch_embed_evt[0].weight.copy_(src_w)

    def encode(
        self,
        img_rgb: torch.Tensor,
        img_evt: torch.Tensor,
    ) -> torch.Tensor:
        img_rgb = torch.nan_to_num(img_rgb, nan=0., posinf=1., neginf=-1.)
        img_evt = torch.nan_to_num(img_evt, nan=0., posinf=1., neginf=-1.)
        f_rgb = self.patch_embed_rgb(img_rgb)   # [B, D, H/P, W/P]
        f_evt = self.patch_embed_evt(img_evt)   # [B, D, H/P, W/P]
        return self.sor(f_rgb, f_evt, phi_motion=None)

    def forward(self, img_rgb: torch.Tensor, img_evt: torch.Tensor) -> torch.Tensor:
        return self.encode(img_rgb, img_evt)