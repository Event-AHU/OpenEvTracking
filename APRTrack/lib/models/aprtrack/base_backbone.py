import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_sep_seg = False

    def finetune_track_hivit(self, cfg):
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.sep_token_flag = True
        self.sep_token_type = 'fb'
        patch_pos_embed = self.absolute_pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size // self.patch_size, self.img_size // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        self.zf_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.zb_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.x_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.zf_pos_embed = trunc_normal_(self.zf_pos_embed, std=.02)
        self.zb_pos_embed = trunc_normal_(self.zb_pos_embed, std=.02)
        self.x_pos_embed = trunc_normal_(self.x_pos_embed, std=.02)

