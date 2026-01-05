import torch
import torch.nn as nn
import torch.nn.functional as F

class EventFeatureWarper(nn.Module):
    def __init__(self, in_channels=768):
        super(EventFeatureWarper, self).__init__()
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=3, padding=1)  # 输出 offset: Δx, Δy
        )

    def forward(self, rgb_feat, event_feat):
        rgb_feat = rgb_feat.permute(0, 3, 1, 2).contiguous()
        event_feat = event_feat.permute(0, 3, 1, 2).contiguous()
        
        B, C, H, W = rgb_feat.shape

        # Step 1: predict offset from RGB and Event
        fused_input = torch.cat([rgb_feat, event_feat], dim=1)  # [B, 2C, H, W]
        offset = self.offset_net(fused_input)                   # [B, 2, H, W]

        # Step 2: create normalized sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=event_feat.device),
            torch.linspace(-1, 1, W, device=event_feat.device),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

        # Step 3: normalize offset to [-1, 1] and add to base grid
        offset_norm = offset.permute(0, 2, 3, 1)  # [B, H, W, 2]
        offset_norm[..., 0] /= (W / 2)            # Δx
        offset_norm[..., 1] /= (H / 2)            # Δy
        sampling_grid = base_grid + offset_norm   # [B, H, W, 2]

        
        # Step 4: warp event_feat using sampling grid
        warped_event_feat = F.grid_sample(
            event_feat, sampling_grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )
        
        return warped_event_feat.permute(0, 2, 3, 1).contiguous(), offset


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, t, h, w = x.size()
        avg = self.avg_pool(x).view(b, c)
        max_ = self.max_pool(x).view(b, c)
        y = self.fc(avg) + self.fc(max_)
        y = y.view(b, c, 1, 1, 1)
        return x * y

class SpatialTemporalAttention3D(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: B, C, T, H, W
        avg = torch.mean(x, dim=1, keepdim=True)  # B,1,T,H,W
        max_ = torch.max(x, dim=1, keepdim=True)[0]  # B,1,T,H,W
        y = torch.cat([avg, max_], dim=1)  # B,2,T,H,W
        attn = self.sigmoid(self.conv(y))
        return x * attn

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, channels=768, kernel_size=3, stride=1, padding=None, use_bn=True, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
        
        # depthwise + pointwise
        self.depthwise = nn.Conv3d(channels, channels, kernel_size, stride, padding, groups=channels, bias=bias)
        self.dw_bn = nn.BatchNorm3d(channels) if use_bn else nn.Identity()
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.pw_bn = nn.BatchNorm3d(channels) if use_bn else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        
        # 分维注意力
        self.channel_attn = ChannelAttention3D(channels)
        self.st_attn = SpatialTemporalAttention3D(channels)

    def forward(self, x):
        # (B,T,H,W,C) -> (B,C,T,H,W)
        x = x.permute(0,4,1,2,3).contiguous()
        x = self.depthwise(x)
        x = self.dw_bn(x)
        x = self.pointwise(x)
        x = self.pw_bn(x)
        x = self.activation(x)
        
        # 细化注意力
        x = self.channel_attn(x)
        x = self.st_attn(x)
        
        # (B,C,T,H,W) -> (B,T,H,W,C)
        x = x.permute(0,2,3,4,1).contiguous()
        return x
