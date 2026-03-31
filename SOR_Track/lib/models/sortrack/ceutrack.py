
import os
import math
import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh

# 统一模型类
class CEUTrack(nn.Module):
    """
    CEUTrack 
    arch_mode 决定 forward 行为
    """
    def __init__(self, backbone, box_head,
                 arch_mode: str = "base",
                 sor_frontend=None,
                 embed_dim: int = None,
                 aux_loss: bool = False,
                 head_type: str = "CENTER"):
        super().__init__()
        self.backbone     = backbone
        self.box_head     = box_head
        self.arch_mode    = arch_mode
        self.sor_frontend = sor_frontend   
        self.aux_loss     = aux_loss
        self.head_type    = head_type
        if head_type in ("CORNER", "CENTER"):
            self.feat_sz_s  = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        if aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        valid_modes = {"base", "base_stem", "sor", "stor", "sor_nostem"}
        if arch_mode not in valid_modes:
            raise ValueError(f"arch_mode must be one of {valid_modes}, got '{arch_mode}'")
        if arch_mode != "base" and sor_frontend is None:
            raise ValueError(f"arch_mode='{arch_mode}' requires sor_frontend to be provided")
        
        # sor/stor 单路 head 的通道投影层
        if arch_mode in ('sor', 'stor', 'sor_nostem'):
            if embed_dim is None:
                raise ValueError(f"arch_mode='{arch_mode}' requires embed_dim")
            self.stor_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        else:
            self.stor_proj = None
        self._f_spatial_prev = None  

    # Forward 路由
    def forward(self,
                template: torch.Tensor,           # [B, 3, Hz, Wz]
                search: torch.Tensor,             # [B, 3, Hx, Wx]
                event_template: torch.Tensor,     # [B, 3, Hz, Wz]
                event_search: torch.Tensor,       # [B, 3, Hx, Wx]
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn: bool = False,
                phi_motion: torch.Tensor = None,
                search_prev=None,                 # [B, 3, Hx, Wx]  上一帧 RGB
                event_search_prev=None            # [B, 3, Hx, Wx]  上一帧 Event
                ):
        if self.arch_mode == "base":
            return self._forward_base(
                template, search, event_template, event_search,
                ce_template_mask, ce_keep_rate, return_last_attn)
        
        elif self.arch_mode == "base_stem":         
            return self._forward_base_stem(
                template, search, event_template, event_search,
                ce_template_mask, ce_keep_rate, phi_motion)
        
        elif self.arch_mode == "sor":
            return self._forward_sor(
                template, search, event_template, event_search,
                ce_template_mask, ce_keep_rate, phi_motion)
        
        elif self.arch_mode == "stor":
            _s_prev   = search_prev        if search_prev        is not None else search
            _es_prev  = event_search_prev  if event_search_prev  is not None else event_search
            return self._forward_stor(
                template, search, event_template, event_search,
                _s_prev, _es_prev,
                ce_template_mask, ce_keep_rate)
        
        elif self.arch_mode == "sor_nostem":
            return self._forward_sor_nostem(
                template, search, event_template, event_search,
                ce_template_mask, ce_keep_rate, phi_motion)
        
    
        else:
            raise NotImplementedError(f"arch_mode '{self.arch_mode}' not implemented")
        
    def _forward_base(self, template, search, event_template, event_search,
                      ce_template_mask, ce_keep_rate, return_last_attn):
        """
        原始 CEUTrack 路径
        """
        x, aux_dict = self.backbone.forward_base(
            z=template, x=search,
            event_z=event_template, event_x=event_search,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
        )
        feat_last = x[-1] if isinstance(x, list) else x
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)
        out['backbone_feat'] = x
        return out
    
    def _forward_sor(self, template, search, event_template, event_search,
                     ce_template_mask, ce_keep_rate, phi_motion):
        """
        SOR 路径
        SORFrontend 输出特征图 -> backbone.forward_stortrack
        """
        z_feat = self.sor_frontend(template, event_template)
        x_feat = self.sor_frontend(search,   event_search)
        x, aux_dict = self.backbone.forward_stortrack(
            z_st=z_feat,
            x_st=x_feat,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
        )
        feat_last = x[-1] if isinstance(x, list) else x
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)
        out['backbone_feat'] = x
        return out
    
    def _forward_stor(
        self,
        template, search, event_template, event_search,
        search_prev, event_search_prev,
        ce_template_mask, ce_keep_rate,):
        """
        STOR 完整路径：
        STORFrontend(6路输入) -> z_feat, x_feat
        -> ViT-CE.forward_stortrack -> Head
        """
        if self.training:
            # 训练时用上一帧数据实时计算 f_spatial_prev
            with torch.no_grad():          # prev 帧不参与当前帧的主梯度
                f_rgb_prev = self.sor_frontend.stem_rgb(search_prev)
                f_evt_prev_stem = self.sor_frontend.stem_evt(event_search_prev)
                prev = self.sor_frontend.sor(f_rgb_prev, f_evt_prev_stem, phi_motion=None)
        else:
            if self._f_spatial_prev is None:
                # 用当前帧自身初始化，t=0 时上一帧=当前帧
                with torch.no_grad():
                    _f_rgb = self.sor_frontend.stem_rgb(search)
                    _f_evt = self.sor_frontend.stem_evt(event_search)
                    self._f_spatial_prev = self.sor_frontend.sor(
                        _f_rgb, _f_evt, phi_motion=None
                    ).detach()
            prev = self._f_spatial_prev
        z_feat, x_feat, x_spatial = self.sor_frontend(
            t_rgb      = template,
            t_evt      = event_template,
            s_rgb_t    = search,
            s_evt_t    = event_search,
            s_rgb_prev = search_prev,
            s_evt_prev = event_search_prev,
            f_spatial_prev=prev
        )    
        if not self.training:
            self._f_spatial_prev = x_spatial.detach()

        x, aux_dict = self.backbone.forward_stortrack(
            z_st             = z_feat,
            x_st             = x_feat,
            ce_template_mask = ce_template_mask,
            ce_keep_rate     = ce_keep_rate,
        )
        feat_last = x[-1] if isinstance(x, list) else x
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def _forward_base_stem(
        self, template, search, event_template, event_search,
        ce_template_mask, ce_keep_rate, phi_motion=None,
    ):
        """
        base_stem 路径：
        SPDStem(RGB+Event 融合) -> ViT-CE.forward_stortrack -> Head
        与 base 区别：预处理由 backbone.patch_embed 换为 SPDStem
        Head 输出结构与 base 相同
        """
        z_feat = self.sor_frontend(template, event_template)   # [B, D, hz, wz]
        x_feat = self.sor_frontend(search,   event_search)     # [B, D, hx, wx]

        x, aux_dict = self.backbone.forward_stortrack(
            z_st=z_feat, x_st=x_feat,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
        )
        feat_last = x[-1] if isinstance(x, list) else x
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)
        return out

    def _forward_sor_nostem(
        self, template, search, event_template, event_search,
        ce_template_mask, ce_keep_rate, phi_motion=None
    ):
        """
        sor_nostem 路径：
        PatchEmbed(双路) + SORModule -> ViT-CE.forward_stortrack -> Head
        """
        z_feat = self.sor_frontend(template, event_template)   # [B, D, hz, wz]
        x_feat = self.sor_frontend(search,   event_search)     # [B, D, hx, wx]

        x, aux_dict = self.backbone.forward_stortrack(
            z_st=z_feat, x_st=x_feat,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
        )
        feat_last = x[-1] if isinstance(x, list) else x
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)
        return out
    
    # Head
    def forward_head(self, cat_feature, gt_score_map=None):
        """
        dual-head: 取搜索区末尾 + 模板区头部，拼接后送入 box_head。
        cat_feature: [B, N_z + N_x, C]
        输出特征:    [B, feat_len_s, 2C]
        """
        if self.arch_mode  in ('base', 'base_stem'):
            enc_opt1 = cat_feature[:, -self.feat_len_s:]
            enc_opt2 = cat_feature[:, :self.feat_len_s]
            enc_opt  = torch.cat([enc_opt1, enc_opt2], dim=-1)
        elif self.arch_mode in ('sor', 'stor', 'sor_nostem'):
            enc_opt_raw = cat_feature[:, -self.feat_len_s:]
            enc_opt     = self.stor_proj(enc_opt_raw)

        opt      = enc_opt.unsqueeze(-1).permute(0, 3, 2, 1).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord     = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            return {'pred_boxes': outputs_coord_new, 'score_map': score_map}
        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord_new = bbox.view(bs, Nq, 4)
            return {
                'pred_boxes' : outputs_coord_new,
                'score_map'  : score_map_ctr,
                'size_map'   : size_map,
                'offset_map' : offset_map,
            }
        else:
            raise NotImplementedError(f"head_type '{self.head_type}' not supported")
        
    def reset_temporal_state(self):
        """
        在每个新序列开始前调用，清空 TMA 时序缓存。
        """
        self._f_spatial_prev = None
        
# 工厂函数
def build_ceutrack(cfg, training: bool = True):
    """
    根据 cfg.MODEL.ARCH_MODE 路由
    ARCH_MODE:
      "base"    -> _build_base
      "sor"     -> _build_sor
      "stor" -> _build_stor
      ---- 消融部分 ----
      "nostem"    -> _build_sor_nostem
    """
    arch_mode = getattr(cfg.MODEL, 'ARCH_MODE', 'base').lower()
    
    use_stem = getattr(cfg.MODEL, 'USE_STEM', True)

    if arch_mode == 'sor' and not use_stem:
        arch_mode = 'sor_nostem'

    dispatch = {
        'base'      : _build_base,
        'base_stem' : _build_base_stem,    # ← 新增
        'sor'       : _build_sor,
        'sor_nostem': _build_sor_nostem,   # ← 重构
        'stor'      : _build_stor,
    }
    if arch_mode not in dispatch:
        raise ValueError(f"Unknown ARCH_MODE: '{arch_mode}'")
    return dispatch[arch_mode](cfg, training)
    
# BASE
def _build_base(cfg, training: bool):
    from lib.models.sortrack.vit_ce import (
        vit_base_patch16_224_ce  as ceu_vit_base,
        vit_large_patch16_224_ce as ceu_vit_large,
    )
    current_dir    = os.path.dirname(os.path.abspath(__file__))
    pretrained_dir = os.path.join(current_dir, '../../../pretrained_models')
    #  预训练路径逻辑
    pretrain_file = cfg.MODEL.PRETRAIN_FILE  # e.g. "mae_pretrain_vit_base.pth"
    is_ceutrack_ckpt = training and pretrain_file and ('CEUTrack' in pretrain_file)
    is_mae_ckpt      = training and pretrain_file and ('CEUTrack' not in pretrain_file)

    # MAE 权重传入 backbone 构建函数（在 _create_vision_transformer 内加载）
    pretrained = os.path.join(pretrained_dir, pretrain_file) if is_mae_ckpt else ''
    btype = cfg.MODEL.BACKBONE.TYPE
    if btype == 'vit_base_patch16_224_ce':
        backbone  = ceu_vit_base(pretrained,
                                 use_event_embed=True, 
                                 drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                 ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                 ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
        embed_dim = backbone.embed_dim

    elif btype == 'vit_large_patch16_224_ce':
        backbone  = ceu_vit_large(pretrained,
                                  use_event_embed=True, 
                                  drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                  ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                  ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
        embed_dim = backbone.embed_dim
    else:
        raise NotImplementedError(f"Backbone {btype} not supported in base mode")
    backbone.finetune_track(cfg=cfg, patch_start_index=1)
    
    hidden_dim = embed_dim * 2
    box_head   = build_box_head(cfg, hidden_dim)
    model = CEUTrack(backbone=backbone, 
                     box_head=box_head,
                     arch_mode='base', 
                     sor_frontend=None, 
                     aux_loss=False, 
                     head_type=cfg.MODEL.HEAD.TYPE)
    _init_event_stem_from_rgb(model)
    
    # CEUTrack 完整 checkpoint 加载
    if is_ceutrack_ckpt:
        ckpt_path = os.path.join(pretrained_dir, pretrain_file)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt['net'], strict=False)
        print(f'[_build_base] Loaded CEUTrack ckpt: {ckpt_path}')
        print(f'  missing={len(missing)}, unexpected={len(unexpected)}')

    return model

# SOR
def _build_sor(cfg, training: bool):
    """
    SOR 路径：SPDStem + SORModule + forward_stortrack
    """
    from lib.models.sortrack.vit_ce import (
        vit_base_patch16_224_ce  as stor_vit_base,
        vit_large_patch16_224_ce as stor_vit_large,
    )
    from lib.models.sortrack.sor_frontend import SORFrontend

    current_dir    = os.path.dirname(os.path.abspath(__file__))
    pretrained_dir = os.path.join(current_dir, '../../../pretrained_models')
    pretrained     = ''
    if training and cfg.MODEL.PRETRAIN_FILE and 'CEUTrack' not in cfg.MODEL.PRETRAIN_FILE:
        pretrained = os.path.join(pretrained_dir, cfg.MODEL.PRETRAIN_FILE)
    btype = cfg.MODEL.BACKBONE.TYPE

    if btype == 'vit_base_patch16_224_ce':
        backbone  = stor_vit_base(pretrained,
                                  use_event_embed=False,  
                                  drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                  ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                  ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
        embed_dim = backbone.embed_dim   # 768
    elif btype == 'vit_large_patch16_224_ce':
        backbone  = stor_vit_large(pretrained,
                                   use_event_embed=False,  
                                   drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                   ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                   ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
        embed_dim = backbone.embed_dim   # 1024
    else:
        raise NotImplementedError(f"Backbone {btype} not supported in sor mode")
    if hasattr(backbone, 'finetune_track'):
        backbone.finetune_track(cfg=cfg, patch_start_index=1)

    # SOR Frontend
    stem_scale       = getattr(cfg.MODEL, 'STEM_SCALE',       4)
    sor_K            = getattr(cfg.MODEL, 'SOR_K',            4)
    reduction_stride = getattr(cfg.MODEL, 'REDUCTION_STRIDE', 4)
    sor_frontend = SORFrontend(
        in_channels=3,
        embed_dim=embed_dim,
        stem_scale=stem_scale,
        sor_K=sor_K,
        reduction_stride=reduction_stride,
    )

    # pos_embed_z/x 初始化
    _init_pos_embed_sor(backbone, cfg, embed_dim, sor_frontend)

    # Head：dual-head，hidden_dim = embed_dim * 2
    hidden_dim = embed_dim * 2
    box_head   = build_box_head(cfg, hidden_dim)
    model = CEUTrack(backbone=backbone, box_head=box_head,
                     arch_mode='sor', sor_frontend=sor_frontend,
                     embed_dim=embed_dim, 
                     aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE)
    if training and cfg.MODEL.PRETRAIN_FILE and 'CEUTrack' in cfg.MODEL.PRETRAIN_FILE:
        ckpt = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt['net'], strict=False)
        print(f'[_build_sor] Loaded CEUTrack ckpt, missing={len(missing)}, '
              f'unexpected={len(unexpected)}')
    return model

# STOR
def _build_stor(cfg, training: bool):
    """
    STOR 完整路径构建：SOR + TMA + GIS。
    """
    from lib.models.sortrack.vit_ce import (
        vit_base_patch16_224_ce  as stor_vit_base,
        vit_large_patch16_224_ce as stor_vit_large,
    )
    from lib.models.sortrack.stor_frontend import STORFrontend
    
    current_dir    = os.path.dirname(os.path.abspath(__file__))
    pretrained_dir = os.path.join(current_dir, '../../../pretrained_models')
    pretrained     = ''
    
    if training and cfg.MODEL.PRETRAIN_FILE and 'CEUTrack' not in cfg.MODEL.PRETRAIN_FILE:
        pretrained = os.path.join(pretrained_dir, cfg.MODEL.PRETRAIN_FILE)
    btype = cfg.MODEL.BACKBONE.TYPE

    if btype == 'vit_base_patch16_224_ce':
        backbone  = stor_vit_base(
            pretrained,
            use_event_embed=False,  
            drop_path_rate = cfg.TRAIN.DROP_PATH_RATE,
            ce_loc         = cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio  = cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )
        embed_dim = backbone.embed_dim   # 768
    elif btype == 'vit_large_patch16_224_ce':
        backbone  = stor_vit_large(
            pretrained,
            use_event_embed=False,  
            drop_path_rate = cfg.TRAIN.DROP_PATH_RATE,
            ce_loc         = cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio  = cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )
        embed_dim = backbone.embed_dim   # 1024
    else:
        raise NotImplementedError(f"Backbone {btype} not supported in stor mode")


    if hasattr(backbone, 'finetune_track'):
        backbone.finetune_track(cfg=cfg, patch_start_index=1)

    #  STORFrontend（SOR + TMA + GIS） 
    stem_scale       = getattr(cfg.MODEL, 'STEM_SCALE',       4)
    sor_K            = getattr(cfg.MODEL, 'SOR_K',            4)
    reduction_stride = getattr(cfg.MODEL, 'REDUCTION_STRIDE', 4)
    tma_groups       = getattr(cfg.MODEL, 'TMA_GROUPS',       4)
    stor_frontend = STORFrontend(
        in_channels      = 3,
        embed_dim        = embed_dim,
        stem_scale       = stem_scale,
        sor_K            = sor_K,
        reduction_stride = reduction_stride,
        tma_groups       = tma_groups,
    )

    #  pos_embed 初始化
    _init_pos_embed_sor(backbone, cfg, embed_dim, stor_frontend)

    #  Head（dual-head，hidden_dim = embed_dim * 2） 
    hidden_dim = embed_dim * 2
    box_head   = build_box_head(cfg, hidden_dim)
    #  组装 
    model = CEUTrack(
        backbone     = backbone,
        box_head     = box_head,
        sor_frontend = stor_frontend,   # STORFrontend 赋值给 sor_frontend 字段
        arch_mode    = 'stor',
        embed_dim = embed_dim,
        aux_loss     = False,
        head_type    = cfg.MODEL.HEAD.TYPE,
    )
    # 从 SOR checkpoint 热启动 
    sor_ckpt_path = getattr(cfg.MODEL, 'SOR_PRETRAIN_FILE', '')
    if training and sor_ckpt_path and os.path.exists(sor_ckpt_path):
        ckpt = torch.load(sor_ckpt_path, map_location='cpu')
        state = ckpt.get('net', ckpt)
        # strict=False：TMA/GIS 权重是新增的，SOR/backbone 权重复用
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[_build_stor] Loaded SOR ckpt: {sor_ckpt_path}')
        print(f'  missing={len(missing)}, unexpected={len(unexpected)}')
        # missing 中应只包含 tma.* 和 gis.* 相关键
        tma_gis_missing = [k for k in missing
                           if 'tma' in k or 'gis' in k]
        other_missing   = [k for k in missing
                           if 'tma' not in k and 'gis' not in k]
        if other_missing:
            print(f'非 TMA/GIS missing keys: {other_missing[:10]}')
    return model

"""消融部分"""
# base + std
def _build_base_stem(cfg, training: bool):
    """
    base_stem：SPDStem 预处理 + 原始 CEUTrack ViT + dual-head
    与 base 的区别：无 patch_embed_event，用 BaseStemFrontend 替代
    """
    from lib.models.sortrack.vit_ce import (
        vit_base_patch16_224_ce  as vit_base,
        vit_large_patch16_224_ce as vit_large,
    )
    from lib.models.sortrack.base_stem_frontend import BaseStemFrontend

    current_dir    = os.path.dirname(os.path.abspath(__file__))
    pretrained_dir = os.path.join(current_dir, '../../../pretrained_models')
    pretrain_file  = cfg.MODEL.PRETRAIN_FILE
    is_mae_ckpt    = training and pretrain_file and 'CEUTrack' not in pretrain_file
    pretrained     = os.path.join(pretrained_dir, pretrain_file) if is_mae_ckpt else ''

    btype = cfg.MODEL.BACKBONE.TYPE
    if btype == 'vit_base_patch16_224_ce':
        backbone  = vit_base(pretrained, use_event_embed=False,
                             drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                             ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                             ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
    elif btype == 'vit_large_patch16_224_ce':
        backbone  = vit_large(pretrained, use_event_embed=False,
                              drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                              ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                              ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
    else:
        raise NotImplementedError(btype)

    embed_dim = backbone.embed_dim
    backbone.finetune_track(cfg=cfg, patch_start_index=1)

    stem_scale       = getattr(cfg.MODEL, 'STEM_SCALE',       4)
    reduction_stride = getattr(cfg.MODEL, 'REDUCTION_STRIDE', 1)

    frontend = BaseStemFrontend(
        in_channels      = 3,
        embed_dim        = embed_dim,
        stem_scale       = stem_scale,
        reduction_stride = reduction_stride,
    )
    _init_pos_embed_sor(backbone, cfg, embed_dim, frontend)

    # base_stem 沿用 dual-head → hidden_dim = embed_dim * 2
    hidden_dim = embed_dim * 2
    box_head   = build_box_head(cfg, hidden_dim)

    return CEUTrack(
        backbone     = backbone,
        box_head     = box_head,
        arch_mode    = 'base_stem',
        sor_frontend = frontend,
        embed_dim    = None,        
        aux_loss     = False,
        head_type    = cfg.MODEL.HEAD.TYPE,
    )

# sor + nostd
def _build_sor_nostem(cfg, training: bool):
    from lib.models.sortrack.vit_ce import (
        vit_base_patch16_224_ce  as vit_base,
        vit_large_patch16_224_ce as vit_large,
    )
    from lib.models.sortrack.sor_nostem_frontend import SORNoStemFrontend

    current_dir    = os.path.dirname(os.path.abspath(__file__))
    pretrained_dir = os.path.join(current_dir, '../../../pretrained_models')
    pretrain_file  = cfg.MODEL.PRETRAIN_FILE
    is_mae_ckpt    = training and pretrain_file and 'CEUTrack' not in pretrain_file
    pretrained     = os.path.join(pretrained_dir, pretrain_file) if is_mae_ckpt else ''

    btype = cfg.MODEL.BACKBONE.TYPE
    if btype == 'vit_base_patch16_224_ce':
        backbone  = vit_base(pretrained, use_event_embed=False,
                             drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                             ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                             ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
    elif btype == 'vit_large_patch16_224_ce':
        backbone  = vit_large(pretrained, use_event_embed=False,
                              drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                              ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                              ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
    else:
        raise NotImplementedError(btype)

    embed_dim  = backbone.embed_dim
    backbone.finetune_track(cfg=cfg, patch_start_index=1)

    sor_K      = getattr(cfg.MODEL, 'SOR_K',    4)
    patch_size = getattr(cfg.MODEL, 'PATCH_SIZE', 16)

    frontend = SORNoStemFrontend(
        in_channels = 3,
        embed_dim   = embed_dim,
        patch_size  = patch_size,
        sor_K       = sor_K,
    )
    # 热启动双路 embed
    frontend.init_from_backbone(backbone.patch_embed)
    _init_pos_embed_sor(backbone, cfg, embed_dim, frontend)

    hidden_dim = embed_dim * 2
    box_head   = build_box_head(cfg, hidden_dim)

    return CEUTrack(
        backbone     = backbone,
        box_head     = box_head,
        arch_mode    = 'sor_nostem',
        sor_frontend = frontend,
        embed_dim    = embed_dim,
        aux_loss     = False,
        head_type    = cfg.MODEL.HEAD.TYPE,
    )

# 工具函数
def _init_pos_embed_sor(backbone, cfg, embed_dim, sor_frontend):
    """仅 SOR 路径使用，初始化 pos_embed_z/x。"""
    hz_z = sor_frontend.get_token_grid(cfg.DATA.TEMPLATE.SIZE)
    hz_x = sor_frontend.get_token_grid(cfg.DATA.SEARCH.SIZE)
    nz   = hz_z * hz_z
    nx   = hz_x * hz_x
    pretrained_path = getattr(cfg.MODEL.BACKBONE, 'PRETRAINED_PATH', '')
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt     = torch.load(pretrained_path, map_location='cpu')
        src      = ckpt.get('model', ckpt)
        if 'pos_embed' in src:
            pe_patch = src['pos_embed'][:, 1:, :]          # [1, 196, D]
            N = pe_patch.shape[1]
            H0 = W0 = int(math.sqrt(N))
            pe_z = _interpolate_pos_embed(pe_patch, (H0, W0), (hz_z, hz_z))
            pe_x = _interpolate_pos_embed(pe_patch, (H0, W0), (hz_x, hz_x))
            print(f'[_init_pos_embed_sor] MAE -> Z:{pe_z.shape} X:{pe_x.shape}')
        else:
            pe_z, pe_x = _rand_pos_embed(nz, embed_dim), _rand_pos_embed(nx, embed_dim)
    else:
        pe_z, pe_x = _rand_pos_embed(nz, embed_dim), _rand_pos_embed(nx, embed_dim)
        print(f'[_init_pos_embed_sor] random init nz={nz} nx={nx}')
    backbone.pos_embed_z = nn.Parameter(pe_z)
    backbone.pos_embed_x = nn.Parameter(pe_x)

def _interpolate_pos_embed(pe, src_grid, tgt_grid):
    import torch.nn.functional as F
    _, N, D  = pe.shape
    Hs, Ws   = src_grid
    Ht, Wt   = tgt_grid
    if (Hs, Ws) == (Ht, Wt):
        return pe
    pe_2d = pe.reshape(1, Hs, Ws, D).permute(0, 3, 1, 2)
    pe_2d = F.interpolate(pe_2d, size=(Ht, Wt), mode='bilinear', align_corners=False)
    return pe_2d.permute(0, 2, 3, 1).reshape(1, Ht * Wt, D)

def _rand_pos_embed(n, d):
    pe = torch.zeros(1, n, d)
    nn.init.trunc_normal_(pe, std=0.02)
    return pe

def _init_event_stem_from_rgb(model):
    """
    用 RGB stem 权重热启动 Event stem
    """
    with torch.no_grad():
        model.backbone.patch_embed_event.proj.weight.copy_(
            model.backbone.patch_embed.proj.weight
        )
        model.backbone.patch_embed_event.proj.bias.copy_(
            model.backbone.patch_embed.proj.bias
        )
    print('[init] Event stem initialized from RGB stem weights')
