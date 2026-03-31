# lib/models/ceutrack/base_stem_frontend.py

import torch
import torch.nn as nn
from lib.models.sortrack.stemnet import SPDStem


class BaseStemFrontend(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        stem_scale: int = 4,
        reduction_stride: int = 4,   # base_stem 默认不再额外降采样
    ):
        super().__init__()
        self.embed_dim        = embed_dim
        self.stem_scale       = stem_scale
        self.reduction_stride = reduction_stride

        self.stem_rgb = SPDStem(in_channels, embed_dim, scale=stem_scale)
        self.stem_evt = SPDStem(in_channels, embed_dim, scale=stem_scale)

        if reduction_stride > 1:
            assert reduction_stride % 2 == 0
            layers, remaining = [], reduction_stride
            while remaining > 1:
                layers += [
                    nn.Conv2d(embed_dim, embed_dim,
                              kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GELU(),
                ]
                remaining //= 2
            layers.append(nn.GroupNorm(32, embed_dim))
            self.reduction = nn.Sequential(*layers)
        else:
            self.reduction = nn.Identity()

    def get_token_grid(self, input_size: int) -> int:
        return (input_size // self.stem_scale) // self.reduction_stride

    def encode(
        self,
        img_rgb: torch.Tensor,
        img_evt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
            feat : [B, D, H', W']
        """
        img_rgb = torch.nan_to_num(img_rgb, nan=0., posinf=1., neginf=-1.)
        img_evt = torch.nan_to_num(img_evt, nan=0., posinf=1., neginf=-1.)
        f = self.stem_rgb(img_rgb) + self.stem_evt(img_evt)
        return self.reduction(f)

    def forward(
        self,
        img_rgb: torch.Tensor,
        img_evt: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(img_rgb, img_evt)