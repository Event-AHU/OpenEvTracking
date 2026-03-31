import math

import torch
import torch.nn.functional as F


def generate_bbox_mask(bbox_mask, bbox):
    b, h, w = bbox_mask.shape
    for i in range(b):
        bbox_i = bbox[i].cpu().tolist()
        bbox_mask[i, int(bbox_i[1]):int(bbox_i[1] + bbox_i[3] - 1), int(bbox_i[0]):int(bbox_i[0] + bbox_i[2] - 1)] = 1
    return bbox_mask


def generate_mask_cond(cfg, bs, device, gt_bbox):
    template_size = cfg.DATA.TEMPLATE.SIZE
    stride = cfg.MODEL.BACKBONE.STRIDE
    template_feat_size = template_size // stride

    if cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'ALL':
        box_mask_z = None
    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_POINT':
        # 中心点索引表：key=feat_size, value=中心像素的 slice
        CTR_POINT_INDEX = {
            7  : slice(3, 4),
            8  : slice(3, 4),
            12 : slice(5, 6),
            14 : slice(6, 7),
            16 : slice(7, 8),   # SOR: 128//8=16
        }
        if template_feat_size not in CTR_POINT_INDEX:
            raise NotImplementedError(
                f"[generate_mask_cond] CTR_POINT 不支持 template_feat_size={template_feat_size}. "
                f"(template_size={template_size}, stride={stride}). "
                f"支持的规格: {list(CTR_POINT_INDEX.keys())}. "
                f"请在 ce_utils.py 的 CTR_POINT_INDEX 中添加对应条目。"
            )
        index      = CTR_POINT_INDEX[template_feat_size]
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_REC':
        # 中心矩形索引表
        CTR_REC_INDEX = {
            7  : slice(3, 4),
            8  : slice(3, 5),
            12 : slice(5, 7),
            16 : slice(6, 10),  # ← SOR 模式: 4×4 中心区域
        }
        if template_feat_size not in CTR_REC_INDEX:
            raise NotImplementedError(
                f"[generate_mask_cond] CTR_REC 不支持 template_feat_size={template_feat_size}. "
                f"支持的规格: {list(CTR_REC_INDEX.keys())}."
            )
        index      = CTR_REC_INDEX[template_feat_size]
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'GT_BOX':
        box_mask_z = torch.zeros([bs, template_size, template_size], device=device)
        box_mask_z = generate_bbox_mask(
            box_mask_z, gt_bbox * template_size
        ).unsqueeze(1).to(torch.float)
        box_mask_z = F.interpolate(
            box_mask_z, scale_factor=1. / stride,
            mode='bilinear', align_corners=False
        )
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
    else:
        raise NotImplementedError(
            f"[generate_mask_cond] 未知 CE_TEMPLATE_RANGE: "
            f"'{cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE}'"
        )
    return box_mask_z


def adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH, base_keep_rate=0.5, max_keep_rate=1, iters=-1):
    if epoch < warmup_epochs:
        return 1
    if epoch >= total_epochs:
        return base_keep_rate
    if iters == -1:
        iters = epoch * ITERS_PER_EPOCH
    total_iters = ITERS_PER_EPOCH * (total_epochs - warmup_epochs)
    iters = iters - ITERS_PER_EPOCH * warmup_epochs
    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) \
        * (math.cos(iters / total_iters * math.pi) + 1) * 0.5

    return keep_rate
