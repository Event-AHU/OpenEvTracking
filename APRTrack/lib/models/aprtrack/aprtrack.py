import os
import torch
from torch import nn
from lib.models.layers.head import build_box_head
from lib.models.aprtrack.hivit import hivit_base
from lib.models.aprtrack.perturbation import AdversarialPerturbationModule
from lib.models.aprtrack.completion import HopfieldCompletion

class APRTrack(nn.Module):
    def __init__(self, backbone, box_head, cfg):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim
        self.box_head = box_head
        self.zfs = 8
        self.xfs = 16
        self.zs = int(self.zfs ** 2)
        self.xs = int(self.xfs ** 2)
        self.apg_cfg = getattr(cfg.MODEL, 'APG', None)
        self.use_apg = bool(getattr(self.apg_cfg, 'ENABLE', False)) if self.apg_cfg else False
        self.apg_warmup_epochs = int(getattr(self.apg_cfg, 'WARMUP_EPOCHS', 10)) if self.apg_cfg else 10
        self.apg = AdversarialPerturbationModule(dim=self.embed_dim, enable_modal=bool(getattr(self.apg_cfg, 'MODAL_ENABLE', True)) if self.apg_cfg else False, enable_spatial=bool(getattr(self.apg_cfg, 'SPATIAL_ENABLE', True)) if self.apg_cfg else False, spatial_mask_ratio_min=float(getattr(self.apg_cfg, 'SPATIAL_MASK_RATIO_MIN', 0.2)) if self.apg_cfg else 0.2, spatial_mask_ratio_max=float(getattr(self.apg_cfg, 'SPATIAL_MASK_RATIO_MAX', 0.4)) if self.apg_cfg else 0.4, spatial_temperature=float(getattr(self.apg_cfg, 'SPATIAL_TEMPERATURE', 1.0)) if self.apg_cfg else 1.0, spatial_anchor_overlap_threshold=float(getattr(self.apg_cfg, 'SPATIAL_ANCHOR_OVERLAP_THRESHOLD', 0.45)) if self.apg_cfg else 0.45, spatial_anchor_penalty_weight=float(getattr(self.apg_cfg, 'SPATIAL_ANCHOR_PENALTY_WEIGHT', 0.5)) if self.apg_cfg else 0.5, size=int(cfg.DATA.SEARCH.SIZE), stride=int(cfg.MODEL.BACKBONE.STRIDE), modal_balance_weight=float(getattr(self.apg_cfg, 'MODAL_BALANCE_WEIGHT', 0.1)) if self.apg_cfg else 0.1, modal_target_probs=list(getattr(self.apg_cfg, 'MODAL_TARGET_PROBS', [0.2, 0.4, 0.4])) if self.apg_cfg else [0.2, 0.4, 0.4], route_probs=list(getattr(self.apg_cfg, 'ROUTE_PROBS', [0.4, 0.3, 0.3])) if self.apg_cfg else [0.4, 0.3, 0.3], route_completion=list(getattr(self.apg_cfg, 'ROUTE_COMPLETION', [False, False, True])) if self.apg_cfg else [False, False, True]) if self.use_apg else None
        self.apg_stats = {}
        self.apg_aux_loss = 0.0
        self.apg_completion_mask = None
        self.temporal_stats = {}
        self.completion_cfg = getattr(cfg.MODEL, 'COMPLETION', None)
        self.completion = None
        if self.completion_cfg and bool(getattr(self.completion_cfg, 'ENABLE', False)):
            self.completion = HopfieldCompletion(dim=self.embed_dim, num_heads=int(getattr(self.completion_cfg, 'NUM_HEADS', 4)), dropout=float(getattr(self.completion_cfg, 'DROPOUT', 0.1)), size=int(cfg.DATA.SEARCH.SIZE), stride=int(cfg.MODEL.BACKBONE.STRIDE), memory_size=int(getattr(self.completion_cfg, 'MEMORY_SIZE', 5)), gate_init_value=float(getattr(self.completion_cfg, 'GATE_INIT_VALUE', -1.0)))

    def reset_memory(self):
        if self.completion is not None:
            self.completion.reset_memory()

    def forward(self, zi, ze, xi, xe, gt_score_map=None, return_completion_aux=False, epoch=None):
        completion_aux = None
        if self.completion is not None:
            xi_before_completion, xe_before_completion = xi, xe
            if return_completion_aux:
                xi, xe, completion_aux = self.completion.retrieve_memory(xi, xe, return_aux=True)
                completion_mask = self.apg_completion_mask
                if completion_mask is not None:
                    completion_mask = completion_mask.to(device=xi.device, dtype=torch.bool).view(-1, 1, 1)
                    xi = torch.where(completion_mask, xi, xi_before_completion)
                    xe = torch.where(completion_mask, xe, xe_before_completion)
            else:
                xi, xe = self.completion.retrieve_memory(xi, xe)
        lens_z = zi.size(1)
        lens_x = xi.size(1)
        lens_cls = 0
        xi = torch.cat([zi, xi], dim=1)
        xe = torch.cat([ze, xe], dim=1)
        xi = self.backbone.pos_drop(xi)
        xe = self.backbone.pos_drop(xe)
        for i, blk in enumerate(self.backbone.blocks[-self.backbone.num_main_blocks:]):
            xi, xe, attn_xi, attn_xe = blk(xi, xe, return_attn=True, i=i, lens_z=lens_z, lens_x=lens_x, lens_cls=lens_cls)
        xi = self.backbone.norm_(xi)
        xe = self.backbone.norm_(xe)
        xi = xi[:, -self.xs:]
        xe = xe[:, -self.xs:]
        x = xi + xe
        B, HW, C = x.size()
        opt_feat = x.permute((0, 2, 1)).contiguous().view(-1, C, self.xfs, self.xfs)
        score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
        out_dict = {'pred_boxes': bbox.view(B, 1, 4), 'score_map': score_map_ctr, 'size_map': size_map, 'offset_map': offset_map}
        return out_dict, attn_xi, attn_xe, completion_aux

def build_aprtrack(cfg, training=True):
    pretrained_path = cfg.MODEL.PRETRAIN_PATH
    pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE) if cfg.MODEL.PRETRAIN_FILE and training else ''

    if cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, attn_drop_rate=cfg.MODEL.BACKBONE.ATTN_DROP_RATE)
    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim

    backbone.finetune_track_hivit(cfg=cfg)

    model = APRTrack(backbone, None, cfg)
    model.box_head = build_box_head(cfg, hidden_dim)
    return model
