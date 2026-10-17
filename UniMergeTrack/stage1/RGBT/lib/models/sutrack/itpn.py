# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Yunjie Tian
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import math
import torch
import torch.nn as nn
from timm.models.registry import register_model
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
        ) if rpe else None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rpe_index is not None:
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, init_values=None,
                 drop=0., attn_drop=0., drop_path=0., rpe=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            if self.attn is not None:
                self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            else:
                self.gamma_1 = None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rpe_index=None, mask=None):
        if self.gamma_2 is None:
            if self.attn is not None:
                x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.attn is not None:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rpe_index, mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, search_size=224, template_size=112, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        search_size = to_2tuple(search_size)
        template_size = to_2tuple(template_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution_search = [search_size[0] // patch_size[0], search_size[1] // patch_size[1]]
        patches_resolution_template = [template_size[0] // patch_size[0], template_size[1] // patch_size[1]]
        self.search_size = search_size
        self.template_size = template_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution_search = self.patch_shape_search = patches_resolution_search
        self.num_patches_search = patches_resolution_search[0] * patches_resolution_search[1]
        self.patches_resolution_template = self.patch_shape_template = patches_resolution_template
        self.num_patches_template = patches_resolution_template[0] * patches_resolution_template[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


# the spatial size are split into 4 patches (then downsample 2x)
# concat them and then reduce them to be of 2x channels.
#
class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# PatchSplit is for upsample
# used in FPN for downstream tasks such as detection/segmentation
class PatchSplit(nn.Module):
    def __init__(self, dim, fpn_dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(
            B, N, H, W, 2, 2, self.fpn_dim
        ).permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, N, 2 * H, 2 * W, self.fpn_dim
        )
        return x


class iTPN(nn.Module):
    def __init__(self, search_size=224, template_size=112, patch_size=16, in_chans=3, num_classes=1000, fpn_dim=256, fpn_depth=2,
                 embed_dim=256, mlp_depth1=3, mlp_depth2=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, init_values=None,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False,
                 num_outs=3, token_type_indicate=False,**kwargs):
        super().__init__()
        assert num_outs in [1, 2, 3, 4, 5]
        self.search_size = search_size
        self.template_size = template_size
        self.token_type_indicate = token_type_indicate
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_outs = num_outs
        self.num_main_blocks = depth
        self.fpn_dim = fpn_dim
        self.mlp_depth1 = mlp_depth1
        self.mlp_depth2 = mlp_depth2
        self.depth = depth

        if self.token_type_indicate:
            self.template_background_token = nn.Parameter(torch.zeros(embed_dim))
            self.template_foreground_token = nn.Parameter(torch.zeros(embed_dim))
            self.search_token = nn.Parameter(torch.zeros(embed_dim))

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            search_size=search_size, template_size=template_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
            norm_layer=norm_layer if self.patch_norm else None)
        self.num_patches_search = self.patch_embed.num_patches_search
        self.num_patches_template = self.patch_embed.num_patches_template
        Hp_t = int(self.num_patches_template**0.5)
        Wp_t = Hp_t
        Hp_s = int(self.num_patches_search**0.5)
        Wp_s = Hp_s
        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches_search+self.num_patches_template, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h_s = torch.arange(Hp_s)
            coords_w_s = torch.arange(Wp_s)
            coords_s = torch.stack(torch.meshgrid([coords_h_s, coords_w_s]))
            coords_flatten_s = torch.flatten(coords_s, 1)
            coords_h_t = torch.arange(Hp_s,Hp_s+Hp_t)
            coords_w_t = torch.arange(Wp_s,Wp_s+Wp_t)
            coords_t = torch.stack(torch.meshgrid([coords_h_t, coords_w_t]))
            coords_flatten_t = torch.flatten(coords_t, 1)
            coords_flatten = torch.cat((coords_flatten_s,coords_flatten_t),dim=1)#2,lt+ls
            # caculate relative
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += (Hp_t+Hp_s - 1)
            relative_coords[:, :, 1] += (Wp_t+Wp_s - 1)
            relative_coords[:, :, 0] *= 2 * (Wp_t+Wp_s) - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, mlp_depth1 + mlp_depth2 + depth))
        self.blocks = nn.ModuleList()

        self.blocks.extend([
            BlockWithRPE(
                Hp_t+Hp_s, mlvl_dims['4'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(mlp_depth1)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['4'], norm_layer))
        self.blocks.extend([
            BlockWithRPE(
                Hp_t+Hp_s, mlvl_dims['8'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(mlp_depth2)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['8'], norm_layer))
        self.blocks.extend([
            BlockWithRPE(
                Hp_t+Hp_s, mlvl_dims['16'], num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(depth)]
        )

        ## FPN PART modules are not needed for classification, we delete them here ##

        if self.num_outs == 1:
            self.fc_norm = norm_layer(self.num_features)
            # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.absolute_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.absolute_pos_embed
        patch_pos_embed = self.absolute_pos_embed
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def get_num_layers(self):
        return self.mlp_depth1 + self.mlp_depth2 + self.depth
    def create_mask(self, image, image_anno):
        height = image.size(2)
        width = image.size(3)

        # Extract bounding box coordinates
        x0 = (image_anno[:, 0] * width).unsqueeze(1)
        y0 = (image_anno[:, 1] * height).unsqueeze(1)
        w = (image_anno[:, 2] * width).unsqueeze(1)
        h = (image_anno[:, 3] * height).unsqueeze(1)

        # Generate pixel indices
        x_indices = torch.arange(width, device=image.device)
        y_indices = torch.arange(height, device=image.device)

        # Create masks for x and y coordinates within the bounding boxes
        x_mask = ((x_indices >= x0) & (x_indices < x0 + w)).float()
        y_mask = ((y_indices >= y0) & (y_indices < y0 + h)).float()

        # Combine x and y masks to get final mask
        mask = x_mask.unsqueeze(1) * y_mask.unsqueeze(2) # (b,h,w)

        return mask

    def prepare_tokens_with_masks(self, template_list, search_list, template_anno_list):
        num_template = len(template_list)
        num_search = len(search_list)

        z = torch.stack(template_list, dim=1)  # (b,n,c,h,w)
        z = z.view(-1, *z.size()[2:])  # (bn,c,h,w)
        x = torch.stack(search_list, dim=1)  # (b,n,c,h,w)
        x = x.view(-1, *x.size()[2:])  # (bn,c,h,w)
        z_anno = torch.stack(template_anno_list, dim=1)  # (b,n,4)
        z_anno = z_anno.view(-1, *z_anno.size()[2:])  # (bn,4)
        if self.token_type_indicate:
            # generate the indicate_embeddings for z
            z_indicate_mask = self.create_mask(z, z_anno)
            z_indicate_mask = z_indicate_mask.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size) # to match the patch embedding
            z_indicate_mask = z_indicate_mask.mean(dim=(3,4)).flatten(1) # elements are in [0,1], float, near to 1 indicates near to foreground, near to 0 indicates near to background
            template_background_token = self.template_background_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            template_foreground_token = self.template_foreground_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            weighted_foreground = template_foreground_token * z_indicate_mask.unsqueeze(-1)
            weighted_background = template_background_token * (1 - z_indicate_mask.unsqueeze(-1))
            z_indicate = weighted_foreground + weighted_background

        z = self.patch_embed(z)
        x = self.patch_embed(x)
        # forward stage1&2
        for blk in self.blocks[:-self.num_main_blocks]:
            z = checkpoint.checkpoint(blk, z) if self.use_checkpoint else blk(z)# bn,c,h,w
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)

        x = x[..., 0, 0, :]  # bn,l,c
        z = z[..., 0, 0, :]

        if self.ape:
            x = x + self.absolute_pos_embed[:, :self.num_patches_search, :]
            z = z + self.absolute_pos_embed[:, self.num_patches_search:, :]

        if self.token_type_indicate:
            # generate the indicate_embeddings for x
            x_indicate = self.search_token.unsqueeze(0).unsqueeze(1).expand(x.size(0), x.size(1), self.embed_dim)
            # add indicate_embeddings to z and x
            x = x + x_indicate
            z = z + z_indicate


        z = z.view(-1, num_template, z.size(-2), z.size(-1))  # b,n,l,c
        z = z.reshape(z.size(0), -1, z.size(-1))  # b,l,c
        x = x.view(-1, num_search, x.size(-2), x.size(-1))
        x = x.reshape(x.size(0), -1, x.size(-1))
        xz = torch.cat([x, z], dim=1)
        return xz

    def forward_features(self, template_list, search_list,template_anno_list):
        xz = self.prepare_tokens_with_masks(template_list, search_list, template_anno_list)
        xz = self.pos_drop(xz)

        rpe_index = None
        if self.rpe:
            rpe_index = self.relative_position_index.view(-1)

        for blk in self.blocks[-self.num_main_blocks:]:
            xz = checkpoint.checkpoint(blk, xz, rpe_index) if self.use_checkpoint else blk(xz, rpe_index)
        return self.fc_norm(xz)

        #############  FPN is only used for downstrean tasks such as detection/segmentation  ##############
        #############             Here we only define model for classification               ##############

    def forward(self, template_list,search_list,template_anno_list):
        xz = self.forward_features(template_list,search_list,template_anno_list)
        # x = self.head(x)
        out = [xz]
        return out
import warnings
def load_pretrained(model, checkpoint, pos_type):
    # adjust position encoding
    state_dict = checkpoint
    pe = state_dict['absolute_pos_embed'].float()
    b_pe, hw_pe, c_pe = pe.shape
    side_pe = int(math.sqrt(hw_pe))
    side_num_patches_search = int(math.sqrt(model.num_patches_search))
    side_num_patches_template = int(math.sqrt(model.num_patches_template))
    pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0,3,1,2])  #b,c,h,w

    def adjust_pe(pe_2D, side_pe, side_new):
        if pos_type == 'index':
            if side_pe < side_new:
                pe_new_2D = nn.functional.interpolate(pe_2D, [side_new, side_new], align_corners=True, mode='bicubic')
                warnings.warn('The resolution is too large, the POS_TYPE has been modified to \'interpolate\'')
            else:
                pe_new_2D = pe_2D[:,:,0:side_new,0:side_new]
            pe_new = torch.flatten(pe_new_2D.permute([0, 2, 3, 1]), 1, 2)
        elif pos_type == 'interpolate':
            pe_new_2D = nn.functional.interpolate(pe_2D, [side_new, side_new], align_corners=True, mode='bicubic')
            pe_new = torch.flatten(pe_new_2D.permute([0, 2, 3, 1]), 1, 2)#b,l,c
        else:
            raise NotImplementedError('The POS_TYPE should be index or interpolate')
        return pe_new

    if side_pe != side_num_patches_search:
        pe_s = adjust_pe(pe_2D, side_pe, side_num_patches_search)
    else:
        pe_s = pe
    if side_pe != side_num_patches_template:
        pe_t = adjust_pe(pe_2D, side_pe, side_num_patches_template)
    else:
        pe_t = pe
    pe_xz = torch.cat((pe_s, pe_t), dim=1)
    state_dict['absolute_pos_embed'] = pe_xz
    auxiliary_keys = ["template_background_token", "template_foreground_token", "search_token"]
    for key in auxiliary_keys:
        if (key in model.state_dict().keys()) and (key not in state_dict.keys()):
            state_dict[key] = model.state_dict()[key]
    for key in state_dict.keys():
        if "relative_position" in key and "cls" not in key:
            state_dict[key] = model.state_dict()[key]
    model.load_state_dict(state_dict, strict=False)
@register_model
def oriitpnb(pretrained=False, pos_type="interpolate",pretrain_type="", **kwargs):
    model = iTPN(
        patch_size=16, embed_dim=512, mlp_depth1=3, mlp_depth2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=4, num_outs=1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_type=pos_type,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu")
        load_pretrained(model,checkpoint,pos_type)
    return model


@register_model
def oriitpnl(pretrained=False, pos_type="interpolate",pretrain_type="", **kwargs):
    model = iTPN(
        patch_size=16, embed_dim=768, mlp_depth1=2, mlp_depth2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=4, num_outs=1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_type=pos_type,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu"
        )
        load_pretrained(model,checkpoint,pos_type)
    return model

