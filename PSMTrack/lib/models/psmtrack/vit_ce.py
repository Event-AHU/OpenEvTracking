import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class FeatureAlignmentStem(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = dim // reduction
        
        self.align = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False)
        )
        
        nn.init.constant_(self.align[-1].weight, 0.)

    def forward(self, x):
        return x + self.align(x)
    
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

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, cur_layer=i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.align_mid = FeatureAlignmentStem(embed_dim, reduction=4)
        self.align_sparse = FeatureAlignmentStem(embed_dim, reduction=4)

        self.halting_classifiers = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 1),
            )
        self.layer = []
        
        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None, ce_template_mask=None, ce_keep_rate=None, act_warmup_ratio=None):
        B = z.shape[0]

        z = self.patch_embed(z)
        x_sparse, x_mid, x_dense = self.patch_embed(x, multi_search=True)

        z = z + self.pos_embed_z
        x_sparse = x_sparse + self.pos_embed_x
        x_mid = x_mid + self.pos_embed_x
        x_dense = x_dense + self.pos_embed_x

        len_z = z.shape[1]
        len_x = x_sparse.shape[1]

        concat_1 = torch.cat([z, x_dense], dim=1)
        for i in range(6):
            concat_1 = self.blocks[i](concat_1, chunk_lens=[len_x], layer=i)

        seq = concat_1

        threshold = 1.0
        halting_score = torch.zeros(B, 1, device=z.device)
        remainders = torch.zeros(B, 1, device=z.device)
        n_updates = torch.zeros(B, 1, device=z.device)
        global_x_out_final = torch.zeros(B, len_x, self.embed_dim, device=z.device)
        p_list = []

        is_warmup = self.training and (act_warmup_ratio is not None) and (act_warmup_ratio == 0.0)
        for i in range(6, 12):
            if i == 6:
                x_mid_aligned = self.align_mid(x_mid)
                seq = torch.cat([seq, x_mid_aligned], dim=1)
            elif i == 10:
                x_sparse_aligned = self.align_sparse(x_sparse)
                seq = torch.cat([seq, x_sparse_aligned], dim=1)
                
            if i < 10:
              chunk_lens=[len_x, len_x]
            else:
              chunk_lens=[len_x, len_x, len_x]
              
            seq = self.blocks[i](seq, mask=None, chunk_lens=chunk_lens, layer=i)

            still_running = (halting_score < threshold).float()
            if still_running.sum() == 0:
                break

            if is_warmup:
                halt_prob = torch.zeros((B, 1), device=seq.device)
            else:
                hx = seq.mean(1)
                halt_prob = torch.sigmoid(self.halting_classifiers(hx))

            if i == len(self.blocks) - 1:
                halt_prob = torch.ones_like(halt_prob)

            halt_prob = halt_prob * still_running

            new_halted = ((halting_score + halt_prob) >= threshold).float() * still_running
            still_running_next = ((halting_score + halt_prob) < threshold).float() * still_running

            update_weights = halt_prob * still_running_next
            remainder = (threshold - halting_score) * new_halted
            remainder = torch.clamp(remainder, min=0.0)

            p = update_weights + remainder
      
            halting_score = halting_score + p
            remainders = remainders + remainder
            n_updates = n_updates + still_running

            normed_seq = self.norm(seq)
            curr_x_dense = normed_seq[:, len_z : len_z + len_x]
            if i < 10:
                curr_x_mid = normed_seq[:, len_z + len_x : len_z + 2 * len_x]
                curr_out = curr_x_dense + curr_x_mid
            else:
                curr_x_mid = normed_seq[:, len_z + len_x : len_z + 2 * len_x]
                curr_x_sparse = normed_seq[:, len_z + 2 * len_x : len_z + 3 * len_x]
                curr_out = curr_x_dense + curr_x_mid + curr_x_sparse
            
            global_x_out_final = global_x_out_final + curr_out * p.unsqueeze(-1)
            p_list.append(p)
            
            if (halting_score >= threshold).all():
                break

        x_out_final = global_x_out_final
        ponder_cost = (n_updates + remainders).mean()

        return x_out_final, ponder_cost

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None, return_last_attn=False, act_warmup_ratio=None):
        x, ponder_cost  = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, act_warmup_ratio=act_warmup_ratio)

        return x, ponder_cost


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            checkpoint["model"] = {k.replace("backbone.", "", 1): v for k, v in checkpoint["net"].items()}
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)

            # missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print("missing_keys:", missing_keys)
            print("unexpected_keys:", unexpected_keys)
            print('Load pretrained model from: ' + pretrained)

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
