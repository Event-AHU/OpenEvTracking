# lib/models/ceutrack/sor_frontend.py 
import torch
import torch.nn as nn
from lib.models.sortrack.stemnet import SPDStem
from lib.models.sortrack.sor_module import SORModule


class SORFrontend(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        stem_scale: int = 4,
        sor_K: int = 4,
        reduction_stride: int = 2,
    ):
        super().__init__()
        self.embed_dim        = embed_dim
        self.stem_scale       = stem_scale
        self.reduction_stride = reduction_stride

        self.stem_rgb = SPDStem(in_channels, embed_dim, scale=stem_scale)
        self.stem_evt = SPDStem(in_channels, embed_dim, scale=stem_scale)
        self.sor      = SORModule(in_channels=embed_dim, K=sor_K)

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
        img_rgb = torch.nan_to_num(img_rgb, nan=0., posinf=1., neginf=-1.)
        img_evt = torch.nan_to_num(img_evt, nan=0., posinf=1., neginf=-1.)
        f_rgb = self.stem_rgb(img_rgb)
        f_evt = self.stem_evt(img_evt)
        f_spatial = self.sor(f_rgb, f_evt, phi_motion=None)
        return self.reduction(f_spatial)

    def forward(self, img_rgb: torch.Tensor, img_evt: torch.Tensor) -> torch.Tensor:
        return self.encode(img_rgb, img_evt)