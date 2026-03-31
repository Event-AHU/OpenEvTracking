import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed, PatchEmbed_event, xcorr_depthwise
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        num_patches = self.patch_embed.num_patches
        # 独立的 patch_embed（Frame 模态，in_chans=3）
        self.patch_embed_event = embed_layer(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # self.pos_embed_event = PatchEmbed_event(in_chans=32, embed_dim=768, kernel_size=4, stride=4)
        # self.pos_embed_event_z = PatchEmbed_event(in_chans=32, embed_dim=768, kernel_size=3, stride=1)
        # attn = CrossAttn(768, 4, 3072, 0.1, 'relu')
        # self.cross_attn = Iter_attn(attn, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)
        self.cat_mode      = 'direct'
        self.add_cls_token = False
        self.add_sep_seg   = False

        if not hasattr(self, 'pos_embed_z'):
            self.register_parameter('pos_embed_z', None)
        if not hasattr(self, 'pos_embed_x'):
            self.register_parameter('pos_embed_x', None)

        self.init_weights(weight_init)

    def forward_features(self, z, x, event_z, event_x,
                         mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # event_z = self.pos_embed_event(event_z)     # [:,:,:,:1000]
        # event_x = self.pos_embed_event(event_x)     # B 768 1024
        
        z = self.patch_embed(z)              # RGB  template 
        x = self.patch_embed(x)              # RGB  search  
        event_z = self.patch_embed_event(event_z)  # Event template
        event_x = self.patch_embed_event(event_x)  # Event search

        event_z += self.pos_embed_z
        event_x += self.pos_embed_x
        z += self.pos_embed_z
        x += self.pos_embed_x

        # attention mask handling   # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, event_z, x, event_x, mode=self.cat_mode)        # 64+64+256+256=640
        # x = combine_tokens(z, x, event_z, event_x, mode=self.cat_mode)        # 64+64+256+256=640
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)
        
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s,
                    mask_x,            # 四路 token 拼接后的 mask（可为 None）
                    ce_template_mask,
                    ce_keep_rate)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
        x = self.norm(x)
        
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new*2]
        x = x[:, lens_z_new*2:]

        x = x[:, :lens_x]   # RGB head
        x = torch.cat([event_x, x], dim=1)
        
        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
        }

        return x, aux_dict

    def forward(self, z, x, event_z, event_x,
                ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, event_z, event_x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict

def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            #  格式自动识别 
            # 优先级：net（CEUTrack完整ckpt）> model（MAE/标准）> 直接是 state_dict
            if "net" in checkpoint:
                # CEUTrack 完整 checkpoint：键名带 "backbone." 前缀，需要剥离
                state_dict = {
                    k.replace("backbone.", "", 1): v
                    for k, v in checkpoint["net"].items()
                }
                print(f'[vit_ce_ceu] Loading from CEUTrack checkpoint (net key)')
            elif "model" in checkpoint:
                # MAE 预训练权重 或 标准 timm checkpoint
                state_dict = checkpoint["model"]
                print(f'[vit_ce_ceu] Loading from MAE/timm checkpoint (model key)')
            else:
                # 文件本身就是 state_dict（少数情况）
                state_dict = checkpoint
                print(f'[vit_ce_ceu] Loading from raw state_dict checkpoint')
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False)
            print(f'  missing_keys ({len(missing_keys)}): '
                  f'{missing_keys[:5]}{"..." if len(missing_keys)>5 else ""}')
            print(f'  unexpected_keys ({len(unexpected_keys)}): '
                  f'{unexpected_keys[:5]}{"..." if len(unexpected_keys)>5 else ""}')
            print(f'  Loaded pretrained model from: {pretrained}')
    return model

def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
