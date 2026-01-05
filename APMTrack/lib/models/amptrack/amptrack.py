"""
Basic amptrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from .encoder import build_encoder
from lib.models.layers.head import build_box_head
# from lib.models.amptrack.vit_ce import hivit_small, hivit_base
# from lib.models.amptrack.vit import vit_base_patch16_224
# from lib.models.amptrack.vit_ce_BACKUPS import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
import torch.nn.functional as F
from lib.models.layers.fftfusion import FFTFusion

class AMPTrack(nn.Module):
    """ This is the base class for amptrack """

    def __init__(self, transformer, box_head, fft_fusion, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.fusion_method = fft_fusion

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                event_template: torch.Tensor,  
                event_search: torch.Tensor,          
                event_template_voxel: torch.Tensor,
                event_search_voxel: torch.Tensor,  
                ):
        
        template = self.fusion_method(template, event_template_voxel)
        search = self.fusion_method(search, event_search_voxel)

        # before feeding into backbone, we need to concat four vectors, or two two concat;
        x = self.backbone(z=template, x=search, 
                          event_template_voxel=event_template_voxel, event_search_voxel=event_search_voxel)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
        
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # output the last 256
        # enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)  768*256
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        ## dual head   768+768)*256
        # enc_opt1 = cat_feature[:, -self.feat_len_s:]
        # enc_opt2 = cat_feature[:, :self.feat_len_s]
        
        # enc_opt1, enc_opt2 = cat_feature.chunk(2, dim=1)
        # enc_opt = torch.cat([enc_opt1, enc_opt2], dim=-1)
        
        enc_opt = cat_feature

        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_amptrack(cfg, training=True):    
    backbone = build_encoder(cfg)
    
    hidden_dim = 512

    box_head = build_box_head(cfg, hidden_dim)

    fft_fusion = FFTFusion()

    model = AMPTrack(
        backbone,
        box_head,
        fft_fusion,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'AMPTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model