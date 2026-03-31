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
        self.cat_mode = 'direct' # 拼接模式
        self.add_cls_token = False # STOR 架构通常不需要 CLS token
        self.add_sep_seg = False   # 是否添加 Segment Embedding


        for _name in ('pos_embed_z', 'pos_embed_x'):
            self._parameters.pop(_name, None)   # 清除已注册的 Parameter
            self.__dict__.pop(_name, None)       # 清除普通属性
            self.register_parameter(_name, None) # 重新注册为可选 Parameter

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        # attention mask handling
        # B, H, W
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

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
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
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x,
                event_z=None, event_x=None,      # 原始 CEUTrack 路径参数（兼容）
                ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):
        """
        统一路由入口：
          - z/x 是 [B,C,H,W] 特征图（SOR路径，已经过 Frontend）
            → 调用 forward_stortrack
          - z/x 是 [B,3,H,W] 原始图像（base路径）
            → 调用 forward_features（需要 PatchEmbed）

        判断依据：pos_embed_z 是否已注册为非 None Parameter
        若 pos_embed_z 存在 → SOR 路径（特征图输入）
        若 pos_embed_z 为 None → base 路径（图像输入，走原始 forward_features）
        """
        if self.pos_embed_z is None:
            raise RuntimeError(
                "[vit_ce.forward] pos_embed_z is None. "
                "SOR路径必须在 build_ceutrack 中初始化 pos_embed_z/x。"
                "如需 base 路径，请使用 vit_ce_ceu.py。"
            )
        x, aux_dict = self.forward_stortrack(
            z, x,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
        )
        return x, aux_dict

    def forward_stortrack(self, z_st, x_st, ce_template_mask=None, ce_keep_rate=None):
        """
        Method - forward_stortrack
        STOR 架构的推理/训练流水线
        
        Args:
            z_st: [B, C, Hz, Wz] (经 SOR 处理后的模板特征)
            x_st: [B, C, Hx, Wx] (经 GIS 处理后的搜索区域特征)
        """
        Bz, Cz, Hz, Wz = z_st.shape
        Bx, Cx, Hx, Wx = x_st.shape

        # 序列化 (Flattening)
        # [B, C, H, W] -> [B, HW, C]
        z = z_st.flatten(2).transpose(1, 2)
        x = x_st.flatten(2).transpose(1, 2)

        if z.shape[0] != Bx:
            z = z.expand(Bx, -1, -1).contiguous()
        # 注入位置编码 (Positional Embedding)
        # 需要确保 z 和 x 的长度与 pos_embed_z/x 匹配
        
        if self.pos_embed_z is not None:
            if z.shape[1] == self.pos_embed_z.shape[1]:
                z = z + self.pos_embed_z
            else:
                # 如果尺寸不匹配（如输入了不同分辨率），需要动态插值 pos_embed
                z = z + self._rescale_pos_embed(self.pos_embed_z, (Hz, Wz))
        else:
            raise ValueError('Pos_embed_z is None, please check the data pipeline')

        if self.pos_embed_x is not None: 
            if x.shape[1] == self.pos_embed_x.shape[1]:
                x = x + self.pos_embed_x
            else:
                x = x + self._rescale_pos_embed(self.pos_embed_x, (Hx, Wx))
        else:
            raise ValueError('Pos_embed_x is None, please check the data pipeline')

        # 如果有分段编码 (Segment Embedding)，在此注入
        if hasattr(self, 'add_sep_seg') and self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed
        # 合并 Tokens (Concatenate)
        # 结果形状: [B, N_z + N_x, C]
        combined_tokens = torch.cat([z, x], dim=1)
        combined_tokens = self.pos_drop(combined_tokens)

        # 准备 Candidate Elimination (CE) 所需的全局索引
        lens_z = z.shape[1]
        lens_x = x.shape[1]
        global_index_z = torch.arange(lens_z, device=x.device).repeat(Bx, 1)
        global_index_x = torch.arange(lens_x, device=x.device).repeat(Bx, 1)
        
        removed_indexes_s = []
        mask_x = None # 默认不使用静态 mask，由 CE 动态计算

        # Transformer Blocks 循环
        curr_tokens = combined_tokens
        for i, blk in enumerate(self.blocks):
            # blk 负责处理跨模态注意力和 token 剔除
            curr_tokens, global_index_z, global_index_x, removed_index_s, attn = \
                blk(curr_tokens, global_index_z, global_index_x, mask_x, ce_template_mask, ce_keep_rate)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
        
        # 归一化与特征还原 (Recovery)
        curr_tokens = self.norm(curr_tokens)
        
        # 将被 CE 剔除掉的 token 用 0 填充回来，以保持输出分辨率固定给 Head
        final_z = curr_tokens[:, :global_index_z.shape[1]]
        final_x_active = curr_tokens[:, global_index_z.shape[1]:]
        
        # 还原 X 的完整序列 (B, Hx*Wx, C)
        final_x = self._recover_x_tokens(final_x_active, global_index_x, lens_x, removed_indexes_s)
        # 重新拼接输出 (Head 预期输入是拼接好的序列)
        out_tokens = torch.cat([final_z, final_x], dim=1)
        last_attn = attn 
        aux_dict = {
            "attn": last_attn,
            "removed_indexes_s": removed_indexes_s,
        }
        return out_tokens, aux_dict
    
    def _recover_x_tokens(self, x_active, global_index_x, total_len, removed_indexes_s):
        """将经 CE 剔除后的 active tokens 恢复到完整的 total_len 长度序列。"""
        B, N_active, C = x_active.shape
        # CE 未激活或无 token 被剔除
        if N_active == total_len:
            return x_active
        # 用 global_index_x（active token 的原始位置索引）做 scatter 还原
        # global_index_x: [B, N_active]，值域 [0, total_len)
        out = torch.zeros(B, total_len, C, device=x_active.device, dtype=x_active.dtype)
        idx = global_index_x.unsqueeze(-1).expand(B, N_active, C).to(torch.int64)
        out.scatter_(dim=1, index=idx, src=x_active)
        return out

    def _rescale_pos_embed(self, pos_embed, new_shape):
        """
        当输入分辨率改变时，动态插值位置编码
        Args:
            pos_embed: [1, N, C]
            new_shape: (H, W)
        """
        B, N, C = pos_embed.shape
        H = W = int(math.sqrt(N))
        # [1, N, C] -> [1, C, H, W]
        pos_embed = pos_embed.reshape(1, H, W, C).permute(0, 3, 1, 2)
        # 插值
        pos_embed = F.interpolate(pos_embed, size=new_shape, mode='bilinear', align_corners=False)
        # [1, C, H_new, W_new] -> [1, N_new, C]
        pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return pos_embed

def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            if "net" in checkpoint:
                state_dict = {
                    k.replace("backbone.", "", 1): v
                    for k, v in checkpoint["net"].items()
                    if k.startswith("backbone.")   
                }
                print(f'[vit_ce] Loading from CEUTrack checkpoint (net key), '
                      f'extracted {len(state_dict)} backbone keys')
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
                print(f'[vit_ce] Loading from MAE/timm checkpoint (model key)')
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                print(f'[vit_ce] Loading from state_dict key checkpoint')
            else:
                state_dict = checkpoint
                print(f'[vit_ce] Loading from raw state_dict checkpoint')

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f'[vit_ce] missing={len(missing)}, unexpected={len(unexpected)}')
       
    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model_kwargs.update(kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model_kwargs.update(kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
