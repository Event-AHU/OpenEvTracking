# lib/test/tracker/ceutrack.py
import math

import numpy as np

from lib.models.sortrack import build_ceutrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import copy

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import cv2
import os
import torch.nn.functional as F
from lib.test.tracker.data_utils import Preprocessor, EventPreprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.test.tracker.sor_visualizer import SORVisualizer


class CEUTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(CEUTrack, self).__init__(params)
        network = build_ceutrack(params.cfg, training=False)
        network.load_state_dict(
            torch.load(self.params.checkpoint, map_location='cpu')['net'],
            strict=True
        )
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.event_norm_mode = getattr(params, 'event_norm_mode', 'imagenet')  # imagenet, event
        self.event_preprocessor = EventPreprocessor(mode=self.event_norm_mode)    
       
        self.state = None

        self.feat_sz = (self.cfg.TEST.SEARCH_SIZE
                        // self.cfg.MODEL.BACKBONE.STRIDE)
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(),
            centered=True
        ).cuda()

        # 使用 OpenCV 保存
        self.debug        = params.debug
        self.use_visdom   = False          
        self.frame_id     = 0
        self.visdom       = None           

        if self.debug:
            # 保存目录：tracker_name/seq_name/
            seq_tag = getattr(params, 'seq_name', 'unknown_seq')
            tracker_tag = getattr(params, 'tracker_name', 'ceutrack')
            param_tag   = getattr(params, 'parameter_name', 'default')
            self.save_dir = os.path.join(
                "debug_vis", tracker_tag, param_tag, seq_tag
            )
            self._sor_vis = SORVisualizer(
                model    = self.network,
                cfg      = self.cfg,
                save_dir = self.save_dir if self.debug else 'vis_output/default',
                vis_every= None,      
            )
            # bbox 保存路径（供离线合并脚本读取）
            self.bbox_save_path = os.path.join(
                self.save_dir, "_pred_bboxes.txt"
            )
            os.makedirs(self.save_dir, exist_ok=True)
            # 清空/创建 bbox 文件
            open(self.bbox_save_path, 'w').close()
            print(f"[CEUTrack debug] Saving to: {self.save_dir}")

        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, event_template, start_frame_idx, info: dict):
        z_patch_arr, resize_factor, z_amask_arr, crop_coor = sample_target(
            image, info['init_bbox'],
            self.params.template_factor,
            output_sz=self.params.template_size
        )
        z_patch_arr_e, resize_factor_e, z_amask_arr_e, crop_coor_e = sample_target(
            event_template, info['init_bbox'],
            self.params.template_factor,
            output_sz=self.params.template_size
        )

        self.z_patch_arr = z_patch_arr

        template       = self.preprocessor.process(z_patch_arr,   z_amask_arr)
        event_template = self.event_preprocessor.process(z_patch_arr_e, z_amask_arr_e)

        with torch.no_grad():
            self.z_dict       = template
            self.z_dict_event = event_template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                info['init_bbox'], resize_factor,
                template.tensors.device
            ).squeeze(1)
            self.box_mask_z = generate_mask_cond(
                self.cfg, 1, template.tensors.device, template_bbox
            )

        self.state    = info['init_bbox']
        self.frame_id = start_frame_idx

        if self.save_all_boxes:
            all_boxes_save = (info['init_bbox']
                              * self.cfg.MODEL.NUM_OBJECT_QUERIES)
            return {"all_boxes": all_boxes_save}

    def track(self, image, event_search, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor, x_amask_arr, _ = sample_target(
            image, self.state, self.params.search_factor,
            output_sz=self.params.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        x_patch_arr_e, resize_factor_e, x_amask_arr_e, _ = sample_target(
            event_search, self.state, self.params.search_factor,
            output_sz=self.params.search_size
        )

        # if self.frame_id <= 5 or (self.frame_id > 125 and self.frame_id < 131):
        #     print(f"[Debug frame {self.frame_id}] Event patch raw: "
        #         f"min={x_patch_arr_e.min()}, max={x_patch_arr_e.max()}, "
        #         f"mean={x_patch_arr_e.mean():.2f}, "
        #         f"zero_ratio={(x_patch_arr_e==0).mean():.3f}")

        # event_search_proc = self.preprocessor.process(x_patch_arr_e, x_amask_arr_e)
        event_search_proc = self.event_preprocessor.process(x_patch_arr_e, x_amask_arr_e)

        # zeros_tensors = event_search_proc.tensors.clone().detach().zero_()
        # event_search_proc = event_search_proc.__class__(zeros_tensors, event_search_proc.mask)
        
        # if self.frame_id <= 5 or (self.frame_id > 125 and self.frame_id < 131):
        #     t = event_search_proc.tensors
        #     print(f"[Debug frame {self.frame_id}] Event tensor normalized: "
        #         f"min={t.min():.3f}, max={t.max():.3f}, "
        #         f"mean={t.mean():.3f}, std={t.std():.3f}")

        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict.tensors,
                search=search.tensors,
                event_template=self.z_dict_event.tensors,
                event_search=event_search_proc.tensors,
                ce_template_mask=self.box_mask_z
            )

        pred_score_map = out_dict['score_map']
        response       = self.output_window * pred_score_map
        pred_boxes     = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map']
        )
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box   = (pred_boxes.mean(dim=0)
                      * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(
            self.map_box_back(pred_box, resize_factor), H, W, margin=10
        )

        if self.debug > 0 and  getattr(self, '_sor_vis', None) is not None \
        and self._sor_vis.should_visualize(self.frame_id):
            # gt_bbox 转换到 search patch 坐标系
            _gt_in_search = self._gt_to_search_coords(
                info.get('gt_bbox') if info else None,
                resize_factor
            )
            self._sor_vis.run(
                frame_id   = self.frame_id,
                rgb_tensor = search.tensors,          # [1,3,Hx,Wx] 已归一化
                evt_tensor = event_search_proc.tensors,
                gt_bbox    = _gt_in_search,
                pred_bbox  = self._pred_in_search(pred_box, resize_factor),
                split      = 'search',
            )

        # if self.frame_id <= 5:
        #     sm   = out_dict['score_map'].squeeze()
        #     szm  = out_dict['size_map'].squeeze()    # (2, H, W) 或 (H, W, 2)
        #     offm = out_dict['offset_map'].squeeze()  # (2, H, W) 或 (H, W, 2)

        #     # score_map 峰值位置
        #     flat_idx = sm.argmax()
        #     peak_y   = (flat_idx // sm.shape[-1]).item()
        #     peak_x   = (flat_idx %  sm.shape[-1]).item()

        #     print('\n--- frame %d raw head output ---' % self.frame_id)
        #     print('  score_map  shape=%s  max=%.4f  peak=(%d,%d)' % (
        #         tuple(sm.shape), sm.max().item(), peak_y, peak_x))
        #     print('  size_map   shape=%s' % str(tuple(szm.shape)))
        #     print('  offset_map shape=%s' % str(tuple(offm.shape)))

        #     # 取峰值点的 w/h 预测值
        #     if szm.dim() == 3:          # (2, H, W)
        #         sw = szm[0, peak_y, peak_x].item()
        #         sh = szm[1, peak_y, peak_x].item()
        #     else:                        # (H, W, 2)
        #         sw = szm[peak_y, peak_x, 0].item()
        #         sh = szm[peak_y, peak_x, 1].item()
        #     print('  size @ peak: w_raw=%.4f  h_raw=%.4f' % (sw, sh))
        #     print('  → scaled w=%.1f  h=%.1f  (×search_size=%.0f / resize_factor=%.4f)' % (
        #         sw * self.params.search_size / resize_factor,
        #         sh * self.params.search_size / resize_factor,
        #         self.params.search_size, resize_factor))
        #     print('  pred_boxes(raw mean):', pred_boxes.mean(dim=0).tolist())

        #  debug 可视化：OpenCV 保存 
        if self.debug:
            self._save_debug_frame(
                image, x_patch_arr, pred_score_map, info
            )
            self._save_bbox_record(info)

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.params.search_size / resize_factor,
                resize_factor
            )
            return {
                "target_bbox": self.state,
                "all_boxes": all_boxes.view(-1).tolist()
            }
        
        # if self.frame_id <= 10:
        #     print(f"[WRITE] frame{self.frame_id}: "
        #           f"x={self.state[0]:.4f} y={self.state[1]:.4f} "
        #           f"w={self.state[2]:.4f} h={self.state[3]:.4f}")
        return {"target_bbox": self.state}

    def _save_bbox_record(self, info):
        """每帧追加写入: frame_id,px,py,pw,ph,gx,gy,gw,gh"""
        gt = info.get('gt_bbox', None) if info else None
        if hasattr(gt, 'tolist'):
            gt = gt.tolist()
        gt_str = ','.join(f'{v:.2f}' for v in gt) if gt else 'None'
        pred_str = ','.join(f'{v:.2f}' for v in self.state)
        
        with open(self.bbox_save_path, 'a') as f:
            f.write(f"{self.frame_id},{pred_str},{gt_str}\n")

    def _save_debug_frame(self, image, x_patch_arr, pred_score_map, info):
        """
        """
        frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        raw_path = os.path.join(self.save_dir, f"{self.frame_id:05d}_raw.jpg")
        cv2.imwrite(raw_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        frame_anno = frame_bgr.copy()
        x1, y1, w, h = [int(v) for v in self.state]
        cv2.rectangle(frame_anno, (x1, y1), (x1+w, y1+h),
                    color=(0, 0, 255), thickness=2)
        if (info is not None
                and 'gt_bbox' in info
                and info['gt_bbox'] is not None):
            gt = info['gt_bbox']
            if hasattr(gt, 'tolist'):
                gt = gt.tolist()
            gx, gy, gw, gh = [int(v) for v in gt]
            cv2.rectangle(frame_anno, (gx, gy), (gx+gw, gy+gh),
                        color=(0, 255, 0), thickness=2)
        #  图例统一放右下角
        # self._draw_legend_corner(frame_anno, [
        #     ((0, 255, 0),   'GT'),
        #     ((0, 0, 255),   'Pred'),
        # ])
        #  缩放 + 拼接 search patch + score map 
        TARGET_H = 320
        scale = TARGET_H / frame_anno.shape[0]
        frame_vis = cv2.resize(
            frame_anno,
            (int(frame_anno.shape[1] * scale), TARGET_H)
        )
        search_bgr = cv2.cvtColor(x_patch_arr, cv2.COLOR_RGB2BGR)
        search_vis = cv2.resize(search_bgr, (TARGET_H, TARGET_H))
        score_np = pred_score_map.squeeze().cpu().float().numpy()
        s_min, s_max = score_np.min(), score_np.max()
        if s_max > s_min:
            score_norm = ((score_np - s_min) / (s_max - s_min) * 255).astype('uint8')
        else:
            score_norm = np.zeros_like(score_np, dtype='uint8')
        score_color = cv2.applyColorMap(
            cv2.resize(score_norm, (TARGET_H, TARGET_H)),
            cv2.COLORMAP_JET
        )
        cv2.putText(score_color, f"max={s_max:.3f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        canvas = cv2.hconcat([frame_vis, search_vis, score_color])
        save_path = os.path.join(self.save_dir, f"{self.frame_id:05d}.jpg")
        cv2.imwrite(save_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor,
                           resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack(
            [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1
        )
    

    def _gt_to_search_coords(self, gt_bbox_orig, resize_factor):
        """
        将原图 GT bbox [x,y,w,h] 转换到 search patch 坐标系。
        复用与 map_box_back 对称的逆变换。
        """
        if gt_bbox_orig is None:
            return None
        if hasattr(gt_bbox_orig, 'tolist'):
            gt_bbox_orig = gt_bbox_orig.tolist()
        gx, gy, gw, gh = gt_bbox_orig
        # search patch 中心 = 上一帧预测框中心
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        half    = 0.5 * self.params.search_size / resize_factor
        # 原图坐标 → search patch 内坐标
        sx = (gx - (cx_prev - half)) * resize_factor
        sy = (gy - (cy_prev - half)) * resize_factor
        sw = gw * resize_factor
        sh = gh * resize_factor
        return [sx, sy, sw, sh]
    
    def _pred_in_search(self, pred_box_mapped, resize_factor):
        """
        将 map_box_back 后的原图坐标 pred_box 再映射回 search 坐标。
        pred_box_mapped: [x,y,w,h] 原图坐标
        """
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        half    = 0.5 * self.params.search_size / resize_factor
        px, py, pw, ph = pred_box_mapped
        sx = (px - (cx_prev - half)) * resize_factor
        sy = (py - (cy_prev - half)) * resize_factor
        sw = pw * resize_factor
        sh = ph * resize_factor
        return [sx, sy, sw, sh]


    def add_hook(self):
        enc_attn_weights = []
        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output:
                    enc_attn_weights.append(output[1])
            )
        self.enc_attn_weights = enc_attn_weights

    def _draw_legend_corner(self, img, items,
                            margin=8, box_w=14, box_h=14,
                            font_scale=0.42, line_h=20):
        """
        在图片右下角绘制图例色块+文字，背景半透明黑色。
        items: [(BGR_color, label), ...]
        """
        n = len(items)
        # 计算图例区域尺寸
        max_text_w = max(
            cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, 1)[0][0]
            for _, label in items
        )
        legend_w = margin + box_w + 6 + max_text_w + margin
        legend_h = margin + n * line_h + margin
        ih, iw = img.shape[:2]
        x0 = iw - legend_w - margin
        y0 = ih - legend_h - margin
        x1 = x0 + legend_w
        y1 = y0 + legend_h
        # 半透明黑色背景
        roi = img[y0:y1, x0:x1]
        bg  = np.zeros_like(roi)
        img[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.35, bg, 0.65, 0)
        # 逐项绘制
        for i, (color, label) in enumerate(items):
            cy = y0 + margin + i * line_h
            # 色块
            cv2.rectangle(img,
                        (x0 + margin,          cy),
                        (x0 + margin + box_w,  cy + box_h),
                        color, -1)
            # 文字
            cv2.putText(img, label,
                        (x0 + margin + box_w + 6, cy + box_h - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (240, 240, 240), 1, cv2.LINE_AA)

def get_tracker_class():
    return CEUTrack