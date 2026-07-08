import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.layers import to_2tuple
from lib.models.aprtrack.base_backbone import BaseBackbone

class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, attn) if return_attn else x

class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., rpe=True, use_checkpoint=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe=rpe) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xi, xe, rpe_index=None, mask=None, return_attn=False, i=0, **kwargs):
        def _inner_forward(xi, xe, return_attn=False):
            if self.attn is not None:
                xi_norm1, xe_norm1 = self.norm1(xi), self.norm1(xe)
                xi_attn, attn_xi = self.attn(xi_norm1, rpe_index, mask, return_attn=True)
                xe_attn, attn_xe = self.attn(xe_norm1, rpe_index, mask, return_attn=True)
                xi = xi + self.drop_path(xi_attn)
                xe = xe + self.drop_path(xe_attn)
                xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
                xe = xe + self.drop_path(self.mlp(self.norm2(xe)))
                return (xi, xe, attn_xi, attn_xe) if return_attn else (xi, xe)
            else:
                xi = xi + self.drop_path(self.mlp(self.norm2(xi)))
                xe = xe + self.drop_path(self.mlp(self.norm2(xe)))
                return xi, xe

        if self.use_checkpoint:
            outputs = checkpoint.checkpoint(_inner_forward, xi, xe, return_attn)
        else:
            outputs = _inner_forward(xi, xe, return_attn=return_attn)

        return outputs

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, xi, xe):
        B, C, H, W = xi.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        xi = self.proj(xi).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        xe = self.proj(xe).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)

        if self.norm is not None:
            xi = self.norm(xi)
            xe = self.norm(xe)
        return xi, xe

class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, xi, xe):
        xi0 = xi[..., 0::2, 0::2, :]
        xi1 = xi[..., 1::2, 0::2, :]
        xi2 = xi[..., 0::2, 1::2, :]
        xi3 = xi[..., 1::2, 1::2, :]
        xe0 = xe[..., 0::2, 0::2, :]
        xe1 = xe[..., 1::2, 0::2, :]
        xe2 = xe[..., 0::2, 1::2, :]
        xe3 = xe[..., 1::2, 1::2, :]
        xi = torch.cat([xi0, xi1, xi2, xi3], dim=-1)
        xe = torch.cat([xe0, xe1, xe2, xe3], dim=-1)
        xi = self.norm(xi)
        xe = self.norm(xe)
        xi = self.reduction(xi)
        xe = self.reduction(xe)
        return xi, xe

class HiViT(BaseBackbone):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=512, depths=[4, 4, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]
        self.norm_ = norm_layer(embed_dim)

        embed_dim = embed_dim // 2 ** (self.num_layers - 1)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.num_features))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))

        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = embed_dim == self.num_features
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for i in range(stage_depth):
                self.blocks.append(BlockWithRPE(Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr), rpe=rpe, norm_layer=norm_layer, use_checkpoint=use_checkpoint))
            if not is_main_stage:
                self.blocks.append(PatchMerge(embed_dim, norm_layer))
                embed_dim *= 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

def load_pretrained(model, pretrained, **kwargs):
    state_dict = torch.load(pretrained, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print('Load pretrained model from: ' + pretrained)
    print('Missing keys:', missing_keys)
    print('Unexpected keys:', unexpected_keys)

def hivit_base(pretrained=False, **kwargs):
    model_kwargs = dict(embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = HiViT(**model_kwargs)
    if pretrained:
        load_pretrained(model, pretrained, **model_kwargs)
    return model
