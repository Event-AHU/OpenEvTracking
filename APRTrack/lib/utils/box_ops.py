import torch
from torchvision.ops.boxes import box_area
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generate_soft_mask(size, stride, bbox):
    if size % stride != 0:
        raise ValueError(f"size={size} must be divisible by stride={stride}")
    B = bbox.size(0)
    device = bbox.device
    dtype = bbox.dtype
    h_tokens = size // stride
    w_tokens = size // stride
    patch_area = float(stride * stride)
    x1 = bbox[:, 0] * size
    y1 = bbox[:, 1] * size
    x2 = (bbox[:, 0] + bbox[:, 2]) * size
    y2 = (bbox[:, 1] + bbox[:, 3]) * size
    patch_x1 = torch.arange(0, size, stride, device=device, dtype=dtype)
    patch_y1 = torch.arange(0, size, stride, device=device, dtype=dtype)
    patch_x2 = patch_x1 + stride
    patch_y2 = patch_y1 + stride
    px1 = patch_x1.view(1, 1, w_tokens)
    px2 = patch_x2.view(1, 1, w_tokens)
    py1 = patch_y1.view(1, h_tokens, 1)
    py2 = patch_y2.view(1, h_tokens, 1)
    bx1 = x1.view(B, 1, 1)
    bx2 = x2.view(B, 1, 1)
    by1 = y1.view(B, 1, 1)
    by2 = y2.view(B, 1, 1)
    inter_w = (torch.minimum(px2, bx2) - torch.maximum(px1, bx1)).clamp(min=0)
    inter_h = (torch.minimum(py2, by2) - torch.maximum(py1, by1)).clamp(min=0)
    soft_mask = ((inter_w * inter_h) / patch_area).reshape(B, -1)
    return soft_mask


def box_iou(boxes1, boxes2):
    """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1)       
    area2 = box_area(boxes2)       
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])         
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])         
    wh = (rb - lt).clamp(min=0)         
    inter = wh[:, 0] * wh[:, 1]        
    union = area1 + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)       
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)         
    area = wh[:, 0] * wh[:, 1]       
    return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
    """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]
