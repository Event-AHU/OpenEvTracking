from lib.models.aprtrack import build_aprtrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import os
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box, box_cxcywh_to_xyxy, box_xyxy_to_xywh
from lib.utils.box_ops import generate_soft_mask

class APRTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(APRTrack, self).__init__(params)
        net = build_aprtrack(params.cfg, training=False)
        net.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.net = net.cuda()
        self.net.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        self.save_all_boxes = params.save_all_boxes
        self.update_interval = self.cfg.TEST.UPDATE_INTERVAL
        self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD
        self.update_count = 0

    def initialize(self, image, event_image, info: dict, idx=0):
        z_patch_arr, event_z_patch_arr, resize_factor, z_amask_arr = sample_target(im=image, eim=event_image,
            target_bb=info['init_bbox'], search_area_factor=self.params.template_factor, output_sz=self.params.template_size)

        self.z_patch_arr = z_patch_arr
        self.event_z_patch_arr = event_z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors
        event_template = self.preprocessor.process(event_z_patch_arr, z_amask_arr).tensors

        template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor, template.device).squeeze(1)
        self.mask_z = generate_soft_mask(size=self.cfg.TEST.TEMPLATE_SIZE, stride=self.cfg.MODEL.BACKBONE.STRIDE, bbox=template_bbox)
        self.net.reset_memory()

        with torch.no_grad():
            if self.net.completion is not None:
                search_patch_arr, event_search_patch_arr, search_resize_factor, search_amask_arr = sample_target(im=image, eim=event_image, target_bb=info['init_bbox'], search_area_factor=self.params.search_factor, output_sz=self.params.search_size)
                search = self.preprocessor.process(search_patch_arr, search_amask_arr).tensors
                event_search = self.preprocessor.process(event_search_patch_arr, search_amask_arr).tensors
                search_bbox = self.transform_bbox_to_crop(info['init_bbox'], search_resize_factor, search.device, crop_type='search').squeeze(1)
                xi, xe = search.unsqueeze(0), event_search.unsqueeze(0)
                B, M, C, H, W = xi.size()
                xi = xi.reshape(-1, C, H, W)
                xe = xe.reshape(-1, C, H, W)
                xi, xe = self.net.backbone.patch_embed(xi, xe)
                for blk in self.net.backbone.blocks[:-self.net.backbone.num_main_blocks]:
                    xi, xe = blk(xi, xe)
                xi = xi.flatten(2)
                xe = xe.flatten(2)
                xi += self.net.backbone.pos_embed_x + self.net.backbone.x_pos_embed
                xe += self.net.backbone.pos_embed_x + self.net.backbone.x_pos_embed
                xi = xi.reshape(B, -1, self.net.embed_dim)
                xe = xe.reshape(B, -1, self.net.embed_dim)
                self.net.completion.store_memory(xi, xe, search_bbox)
            self.mask_sz = self.mask_z
            self.mask_dz = self.mask_z
            zi, ze = template.unsqueeze(0), event_template.unsqueeze(0)
            B, M, C, H, W = zi.size()
            zi = zi.reshape(-1, C, H, W)
            ze = ze.reshape(-1, C, H, W)
            zi, ze = self.net.backbone.patch_embed(zi, ze)
            for blk in self.net.backbone.blocks[:-self.net.backbone.num_main_blocks]:
                zi, ze = blk(zi, ze)
            zi = zi.flatten(2)
            ze = ze.flatten(2)
            zi += self.net.backbone.pos_embed_z
            ze += self.net.backbone.pos_embed_z
            mask = self.mask_sz.reshape(-1, self.net.zs, 1)
            z_pos_embed = self.net.backbone.zf_pos_embed * mask + self.net.backbone.zb_pos_embed * (1 - mask)
            zi += z_pos_embed
            ze += z_pos_embed
            self.szi = zi.reshape(B, -1, self.net.embed_dim)
            self.sze = ze.reshape(B, -1, self.net.embed_dim)
            self.dzi, self.dze = self.szi, self.sze
        self.state = info['init_bbox']
        self.frame_id = idx
        self.update_count = 0

    def track(self, image, event_image, info: dict = None):
        self.frame_id += 1
        H, W, _ = image.shape
        x_patch_arr, event_x_patch_arr, resize_factor, x_amask_arr = sample_target(im=image, eim=event_image,
            target_bb=self.state, search_area_factor=self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr).tensors
        event_search = self.preprocessor.process(event_x_patch_arr, x_amask_arr).tensors

        with torch.no_grad():
            zi = torch.cat([self.szi, self.dzi], dim=1)
            ze = torch.cat([self.sze, self.dze], dim=1)
            xi, xe = search.unsqueeze(0), event_search.unsqueeze(0)
            B, M, C, Hx, Wx = xi.size()
            xi = xi.reshape(-1, C, Hx, Wx)
            xe = xe.reshape(-1, C, Hx, Wx)
            xi, xe = self.net.backbone.patch_embed(xi, xe)
            for blk in self.net.backbone.blocks[:-self.net.backbone.num_main_blocks]:
                xi, xe = blk(xi, xe)
            xi = xi.flatten(2)
            xe = xe.flatten(2)
            xi += self.net.backbone.pos_embed_x + self.net.backbone.x_pos_embed
            xe += self.net.backbone.pos_embed_x + self.net.backbone.x_pos_embed
            xi = xi.reshape(B, -1, self.net.embed_dim)
            xe = xe.reshape(B, -1, self.net.embed_dim)
            xi_ori, xe_ori = xi, xe
            out_dict, attn_xi, attn_xe, completion_aux = self.net(zi, ze, xi, xe, None)

        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.net.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map']).view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        pred_score = response.max().item()

        if self.frame_id % self.update_interval == 0 and pred_score > self.update_threshold:
            z_patch_arr, event_z_patch_arr, resize_factor, z_amask_arr = sample_target(im=image, eim=event_image, target_bb=self.state, search_area_factor=self.params.template_factor, output_sz=self.params.template_size)
            template = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors
            event_template = self.preprocessor.process(event_z_patch_arr, z_amask_arr).tensors
            template_bbox = self.transform_bbox_to_crop(self.state, resize_factor, template.device).squeeze(1)
            self.mask_dz = generate_soft_mask(size=self.cfg.TEST.TEMPLATE_SIZE, stride=self.cfg.MODEL.BACKBONE.STRIDE, bbox=template_bbox)
            with torch.no_grad():
                zi, ze = template.unsqueeze(0), event_template.unsqueeze(0)
                B, M, C, Ht, Wt = zi.size()
                zi = zi.reshape(-1, C, Ht, Wt)
                ze = ze.reshape(-1, C, Ht, Wt)
                zi, ze = self.net.backbone.patch_embed(zi, ze)
                for blk in self.net.backbone.blocks[:-self.net.backbone.num_main_blocks]:
                    zi, ze = blk(zi, ze)
                zi = zi.flatten(2)
                ze = ze.flatten(2)
                zi += self.net.backbone.pos_embed_z
                ze += self.net.backbone.pos_embed_z
                mask = self.mask_dz.reshape(-1, self.net.zs, 1)
                z_pos_embed = self.net.backbone.zf_pos_embed * mask + self.net.backbone.zb_pos_embed * (1 - mask)
                zi += z_pos_embed
                ze += z_pos_embed
                self.dzi = zi.reshape(B, -1, self.net.embed_dim)
                self.dze = ze.reshape(B, -1, self.net.embed_dim)
            self.update_count += 1

        if self.net.completion is not None and pred_score > self.update_threshold:
            memory_bbox = box_xyxy_to_xywh(box_cxcywh_to_xyxy(pred_boxes))
            self.net.completion.store_memory(xi_ori, xe_ori, memory_bbox)

        return {"target_bbox": self.state, "response": response}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

def get_tracker_class():
    return APRTrack
