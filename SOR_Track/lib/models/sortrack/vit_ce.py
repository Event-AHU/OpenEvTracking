# lib/models/ceutrack/vit_ce_unified.py
"""
VisionTransformerCE

两条独立的前向路径:
  Path-A  forward_base(z, x, event_z, event_x)
          原始 CEUTrack 四路 Token 拼接路径
          调用方:_forward_base, _forward_sor_nostem

  Path-B  forward_stortrack(z_st, x_st)
          SOR/STOR 特征图直接输入路径
          调用方:_forward_sor, _forward_stor
"""

import math
import logging
from functools import partial

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
    """
    ViT-CE

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
        use_event_embed (bool):independent(For path A)/dependent(For path B) instance of patch_embed_event
    """

    def __init__(
        self,
        img_size=224, patch_size=16, in_chans=3,
        num_classes=1000, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=True,
        representation_size=None, distilled=False,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
        weight_init='',
        ce_loc=None, ce_keep_ratio=None,
        use_event_embed: bool = True,
    ):
        super().__init__()

        # 基础属性 
        self.img_size   = to_2tuple(img_size) if not isinstance(img_size, tuple) else img_size
        self.patch_size = patch_size
        self.in_chans   = in_chans
        self.num_classes     = num_classes
        self.num_features    = self.embed_dim = embed_dim
        self.num_tokens      = 2 if distilled else 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer  = act_layer  or nn.GELU

        #  Patch Embedding 
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Event embed:仅 Path-A（base / sor_nostem）需要
        if use_event_embed:
            self.patch_embed_event = embed_layer(
                img_size=img_size, patch_size=patch_size,
                in_chans=in_chans, embed_dim=embed_dim,
            )
        else:
            self.patch_embed_event = None  # Path-B 不需要

        #  标准 ViT 参数 
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop   = nn.Dropout(p=drop_rate)

        #  Transformer Blocks 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.ce_loc = ce_loc
        ce_index = 0
        blocks = []
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            blocks.append(CEBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                keep_ratio_search=ce_keep_ratio_i,
            ))
        self.blocks = nn.Sequential(*blocks)
        self.norm   = norm_layer(embed_dim)

        #  Token 组织配置 
        self.cat_mode      = 'direct'
        self.add_cls_token = False
        self.add_sep_seg   = False

        #  SOR/STOR 路径位置编码
        for _name in ('pos_embed_z', 'pos_embed_x'):
            if _name in self._parameters:
                # 已是 Parameter 保留
                pass
            elif hasattr(self, _name):
                # 父类以普通属性存在，先删除再注册占位符
                object.__delattr__(self, _name)
                self.register_parameter(_name, None)
            else:
                # 完全不存在，正常注册
                self.register_parameter(_name, None)

        self.init_weights(weight_init)

    # Path-A:原始 CEUTrack 四路 Token 路径
    def forward_base(
        self, z, x, event_z, event_x,
        ce_template_mask=None, ce_keep_rate=None,
        return_last_attn=False,
    ):
        """
        Path-A 入口

        输入:
            z, x           : [B, 3, H, W]  RGB 图像
            event_z, event_x: [B, 3, H, W]  Event 帧

        输出:
            out_tokens : [B, 2*lens_x, D]  拼接后的搜索区特征
            aux_dict
        """
        assert self.patch_embed_event is not None, (
            "forward_base 需要 patch_embed_event，"
            "请在构造时传入 use_event_embed=True"
        )
        assert self.pos_embed_z is not None, (
            "pos_embed_z 未初始化，请确认 finetune_track 已被调用"
        )

        B = x.shape[0]

        # Patch Embedding
        z       = self.patch_embed(z)
        x       = self.patch_embed(x)
        event_z = self.patch_embed_event(event_z)
        event_x = self.patch_embed_event(event_x)

        # 位置编码注入
        z       += self.pos_embed_z
        event_z += self.pos_embed_z
        x       += self.pos_embed_x
        event_x += self.pos_embed_x

        lens_z = self.pos_embed_z.shape[1]   # e.g. 64
        lens_x = self.pos_embed_x.shape[1]   # e.g. 256

        # Token 拼接:[z | evt_z | x | evt_x]，共 (2*lens_z + 2*lens_x) tokens
        combined = combine_tokens(z, event_z, x, event_x, mode=self.cat_mode)
        combined = self.pos_drop(combined)

        # CE 全局索引
        global_index_t = torch.arange(lens_z, device=x.device).repeat(B, 1).float()
        global_index_s = torch.arange(lens_x, device=x.device).repeat(B, 1).float()

        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            combined, global_index_t, global_index_s, removed_index_s, attn = \
                blk(combined, global_index_t, global_index_s,
                    None, ce_template_mask, ce_keep_rate)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        combined = self.norm(combined)

        # 特征提取:取两路搜索区特征拼接送 Head
        lens_z_new = global_index_t.shape[1]
        # combined:[z_active | evt_z_active | x_active | ...]
        x_out     = combined[:, lens_z_new * 2:][:, :lens_x]   # 跳过模板部分，取 RGB-x 的前 lens_x 个 token
        out_tokens = torch.cat([event_x, x_out], dim=1)  # [B, 2*lens_x, D]

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
        }
        return out_tokens, aux_dict

    
    # Path-B:SOR/STOR 特征图直接输入路径
    def forward_stortrack(
        self, z_st, x_st,
        ce_template_mask=None, ce_keep_rate=None,
    ):
        """
        Path-B 入口

        输入:
            z_st : [B, C, Hz, Wz]  经 SORFrontend 处理的模板特征图
            x_st : [B, C, Hx, Wx]  经 GIS 处理的搜索区特征图

        输出:
            out_tokens : [B, lens_z + lens_x, D]
            aux_dict
        """
        assert self.pos_embed_z is not None, (
            "forward_stortrack 需要 pos_embed_z，"
            "请确认 _init_pos_embed_sor 已被调用"
        )

        Bz, _, Hz, Wz = z_st.shape
        Bx, _, Hx, Wx = x_st.shape

        # 序列化
        z = z_st.flatten(2).transpose(1, 2)   # [B, Hz*Wz, C]
        x = x_st.flatten(2).transpose(1, 2)   # [B, Hx*Wx, C]

        # batch 对齐
        if Bz != Bx:
            z = z.expand(Bx, -1, -1).contiguous()

        # 位置编码
        z = z + self._get_pos_embed(self.pos_embed_z, (Hz, Wz))
        x = x + self._get_pos_embed(self.pos_embed_x, (Hx, Wx))

        lens_z = z.shape[1]
        lens_x = x.shape[1]

        combined = torch.cat([z, x], dim=1)
        combined = self.pos_drop(combined)

        global_index_z = torch.arange(lens_z, device=x.device).repeat(Bx, 1).float()
        global_index_x = torch.arange(lens_x, device=x.device).repeat(Bx, 1).float()

        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            combined, global_index_z, global_index_x, removed_index_s, attn = \
                blk(combined, global_index_z, global_index_x,
                    None, ce_template_mask, ce_keep_rate)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        combined = self.norm(combined)

        # 还原完整 x 序列
        final_z       = combined[:, :global_index_z.shape[1]]
        final_x_active = combined[:, global_index_z.shape[1]:]
        final_x       = self._recover_x_tokens(
            final_x_active, global_index_x, lens_x, removed_indexes_s
        )

        out_tokens = torch.cat([final_z, final_x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
        }
        return out_tokens, aux_dict

    
    # 统一 forward 入口
    def forward(
        self, z, x,
        event_z=None, event_x=None,
        ce_template_mask=None, ce_keep_rate=None,
        return_last_attn=False,
        # Path-B 专用参数（由 forward_stortrack 直接调用时不走这里）
    ):
        """
        路由入口

        规则:
          event_z/event_x 非 None/None  -> Path-A/Path-B
        """
        if event_z is not None and event_x is not None:
            return self.forward_base(
                z, x, event_z, event_x,
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
                return_last_attn=return_last_attn,
            )
        else:
            return self.forward_stortrack(
                z, x,
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
            )

    
    # 工具方法
    def _get_pos_embed(self, pos_embed: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
        """
        获取位置编码，必要时插值对齐空间尺寸

        Args:
            pos_embed    : [1, N, D]
            spatial_shape: (H, W) 目标空间尺寸

        Returns:
            [1, H*W, D]
        """
        H, W = spatial_shape
        N    = pos_embed.shape[1]
        H0   = W0 = int(math.sqrt(N))

        if H0 == H and W0 == W:
            return pos_embed   

        # 双线性插值
        pe = pos_embed.reshape(1, H0, W0, -1).permute(0, 3, 1, 2)  # [1,D,H0,W0]
        pe = F.interpolate(pe, size=(H, W), mode='bilinear', align_corners=False)
        pe = pe.permute(0, 2, 3, 1).flatten(1, 2)                   # [1,H*W,D]
        return pe

    def _recover_x_tokens(
        self,
        x_active: torch.Tensor,
        global_index_x: torch.Tensor,
        total_len: int,
        removed_indexes_s: list,
    ) -> torch.Tensor:
        """
        将 CE 剔除后的 active tokens 通过 scatter 还原到完整长度

        Args:
            x_active       : [B, N_active, D]
            global_index_x : [B, N_active]  active token 的原始位置索引
            total_len      : 完整序列长度 lens_x
            removed_indexes_s: CE 各层记录的剔除索引（用于调试，不参与计算）

        Returns:
            [B, total_len, D]  剔除位置填 0
        """
        B, N_active, D = x_active.shape
        if N_active == total_len:
            return x_active

        out = torch.zeros(B, total_len, D,
                          device=x_active.device, dtype=x_active.dtype)
        idx = global_index_x.long().unsqueeze(-1).expand(B, N_active, D)
        out.scatter_(dim=1, index=idx, src=x_active)
        return out


# 工厂函数
def _create_vision_transformer(
    pretrained: str = '',
    use_event_embed: bool = True,
    **kwargs,
) -> VisionTransformerCE:
    """
    统一工厂函数

    Args:
        pretrained      : 权重文件路径，空字符串表示随机初始化
        use_event_embed : 是否实例化 patch_embed_event
    """
    model = VisionTransformerCE(use_event_embed=use_event_embed, **kwargs)

    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')

        # 识别 checkpoint 
        if 'net' in checkpoint:
            state_dict = {
                k[len('backbone.'):]: v
                for k, v in checkpoint['net'].items()
                if k.startswith('backbone.')
            }
            _logger.info(f'[vit_ce_unified] CEUTrack ckpt -> {len(state_dict)} backbone keys')
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            _logger.info('[vit_ce_unified] MAE/timm ckpt (model key)')
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            _logger.info('[vit_ce_unified] state_dict key ckpt')
        else:
            state_dict = checkpoint
            _logger.info('[vit_ce_unified] raw state_dict ckpt')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        _logger.info(f'  missing={len(missing)}, unexpected={len(unexpected)}')
        if missing:
            _logger.debug(f'  missing keys: {missing}')
        if unexpected:
            _logger.debug(f'  unexpected keys: {unexpected}')

    return model


def vit_base_patch16_224_ce(pretrained='', use_event_embed=True, **kwargs):
    """ViT-Base (B/16)"""
    return _create_vision_transformer(
        pretrained=pretrained,
        use_event_embed=use_event_embed,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        **kwargs,
    )


def vit_large_patch16_224_ce(pretrained='', use_event_embed=True, **kwargs):
    """ViT-Large (L/16)"""
    return _create_vision_transformer(
        pretrained=pretrained,
        use_event_embed=use_event_embed,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        **kwargs,
    )