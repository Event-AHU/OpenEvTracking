# lib/train/actors/ceutrack.py
from .base_actor import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate


class CEUTrackActor(BaseActor):
    """Actor for training CEUTrack models (base / base_stem / sor / sor_nostem / stor)"""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings    = settings
        self.bs          = self.settings.batchsize
        self.cfg         = cfg
        self.arch_mode   = getattr(cfg.MODEL, 'ARCH_MODE', 'base').lower()
        self.effective_stride = self._compute_effective_stride(cfg)
        print(f"[CEUTrackActor] arch_mode={self.arch_mode}, "
              f"effective_stride={self.effective_stride}")

    
    #  stride 计算                                                         #
    
    def _compute_effective_stride(self, cfg) -> int:
        """
        根据 arch_mode 计算有效步长并做一致性校验
        """
        backbone_stride = cfg.MODEL.BACKBONE.STRIDE

        # PatchEmbed 路径
        if self.arch_mode in ('base', 'sor_nostem'):
            return backbone_stride

        # SPDStem 路径
        if self.arch_mode in ('base_stem', 'sor', 'stor'):
            stem_scale       = getattr(cfg.MODEL, 'STEM_SCALE',       4)
            reduction_stride = getattr(cfg.MODEL, 'REDUCTION_STRIDE', 4)
            effective        = stem_scale * reduction_stride
            assert effective == backbone_stride, (
                f"[CEUTrackActor] ARCH_MODE='{self.arch_mode}': "
                f"STEM_SCALE({stem_scale}) × REDUCTION_STRIDE({reduction_stride}) "
                f"= {effective} != MODEL.BACKBONE.STRIDE={backbone_stride}.\n"
                f"  → 请检查 yaml 中 STEM_SCALE / REDUCTION_STRIDE / BACKBONE.STRIDE 是否对齐。"
            )
            return effective

        raise ValueError(
            f"[CEUTrackActor] _compute_effective_stride: "
            f"unknown arch_mode='{self.arch_mode}'"
        )

    
    #  数据解包工具                                                         #
    @staticmethod
    def _unpack_tensor(data, key: str) -> torch.Tensor:
        """
        从 data[key] 中取第一个样本并展平 batch 维。
        data[key] shape: [num_template/search, B, C, H, W]
        返回: [B, C, H, W]
        """
        t = data[key]
        return t[0].view(-1, *t.shape[2:])

    def _unpack_common(self, data: dict) -> dict:
        """
        所有模式共用的4路输入解包 + CE mask 计算。

        Returns dict with keys:
            template_img, search_img, event_template_img, event_search_img,
            box_mask_z, ce_keep_rate
        """
        template_img       = self._unpack_tensor(data, 'template_images')
        search_img         = self._unpack_tensor(data, 'search_images')
        event_template_img = self._unpack_tensor(data, 'template_event_images')
        event_search_img   = self._unpack_tensor(data, 'search_event_images')

        # NaN 清洗
        template_img       = torch.nan_to_num(template_img,       nan=0., posinf=1., neginf=-1.)
        search_img         = torch.nan_to_num(search_img,         nan=0., posinf=1., neginf=-1.)
        event_template_img = torch.nan_to_num(event_template_img, nan=0., posinf=1., neginf=-1.)
        event_search_img   = torch.nan_to_num(event_search_img,   nan=0., posinf=1., neginf=-1.)

        # CE mask
        box_mask_z   = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(
                self.cfg,
                template_img.shape[0],
                template_img.device,
                data['template_anno'][0],
            )
            ce_keep_rate = adjust_keep_rate(
                data['epoch'],
                warmup_epochs  = self.cfg.TRAIN.CE_START_EPOCH,
                total_epochs   = (self.cfg.TRAIN.CE_START_EPOCH
                                  + self.cfg.TRAIN.CE_WARM_EPOCH),
                ITERS_PER_EPOCH = 1,
                base_keep_rate  = self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0],
            )

        return dict(
            template_img       = template_img,
            search_img         = search_img,
            event_template_img = event_template_img,
            event_search_img   = event_search_img,
            box_mask_z         = box_mask_z,
            ce_keep_rate       = ce_keep_rate,
        )

    
    #  forward_pass 按 arch_mode 分发                                      
    def forward_pass(self, data: dict) -> dict:
        """
        arch_mode 路由表：
          base / base_stem / sor / sor_nostem  →  _forward_4path（4路输入，无 prev 帧）
          stor                                 →  _forward_stor（6路输入，含 prev 帧）
        """
        if self.arch_mode in ('base', 'base_stem', 'sor', 'sor_nostem'):
            return self._forward_4path(data)
        elif self.arch_mode == 'stor':
            return self._forward_stor(data)
        else:
            raise ValueError(
                f"[CEUTrackActor] Unknown arch_mode='{self.arch_mode}'. "
                f"Expected one of: 'base', 'base_stem', 'sor', 'sor_nostem', 'stor'."
            )

    def _forward_4path(self, data: dict) -> dict:
        """
        4路输入路径（base / base_stem / sor / sor_nostem）
        无 prev 帧，接口完全一致。
        """
        inp = self._unpack_common(data)
        return self.net(
            template         = inp['template_img'],
            search           = inp['search_img'],
            event_template   = inp['event_template_img'],
            event_search     = inp['event_search_img'],
            ce_template_mask = inp['box_mask_z'],
            ce_keep_rate     = inp['ce_keep_rate'],
            return_last_attn = False,
        )

    def _forward_stor(self, data: dict) -> dict:
        """
        6路输入路径（stor）：追加 prev 帧的 RGB + Event。
        """
        inp = self._unpack_common(data)

        search_prev_img       = self._unpack_tensor(data, 'search_images_prev')
        event_search_prev_img = self._unpack_tensor(data, 'search_event_images_prev')

        search_prev_img       = torch.nan_to_num(search_prev_img,       nan=0., posinf=1., neginf=-1.)
        event_search_prev_img = torch.nan_to_num(event_search_prev_img, nan=0., posinf=1., neginf=-1.)

        return self.net(
            template          = inp['template_img'],
            search            = inp['search_img'],
            event_template    = inp['event_template_img'],
            event_search      = inp['event_search_img'],
            ce_template_mask  = inp['box_mask_z'],
            ce_keep_rate      = inp['ce_keep_rate'],
            return_last_attn  = False,
            search_prev       = search_prev_img,
            event_search_prev = event_search_prev_img,
        )
    
    #  __call__                                                            
    def __call__(self, data: dict):
        out_dict     = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    
    #  Loss 计算                                                           
    def compute_losses(self, pred_dict: dict, gt_dict: dict, return_status: bool = True):
        device  = pred_dict['score_map'].device
        gt_bbox = gt_dict['search_anno'][-1].to(device)

        gt_gaussian_maps = generate_heatmap(
            gt_dict['search_anno'],
            self.cfg.DATA.SEARCH.SIZE,
            self.effective_stride,
        )
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1).to(device)

        # 前3个 iter 做 heatmap 健康检查
        if not hasattr(self, '_heatmap_checked'):
            self._heatmap_checked = 0
        if self._heatmap_checked < 3:
            hm_max   = gt_gaussian_maps.max().item()
            hm_sum   = gt_gaussian_maps.sum().item()
            sm_shape = pred_dict['score_map'].shape
            print(f"[HeatmapCheck iter={self._heatmap_checked}] "
                  f"gt_bbox[0]={gt_bbox[0].tolist()} "
                  f"hm_max={hm_max:.4f} hm_sum={hm_sum:.4f} "
                  f"score_map.shape={sm_shape} "
                  f"gt_map.shape={gt_gaussian_maps.shape}")
            if hm_max < 0.01:
                raise ValueError(
                    f"[CEUTrackActor] GT heatmap near-zero (max={hm_max:.6f})! "
                    f"gt_bbox[0]={gt_bbox[0].tolist()}, "
                    f"SEARCH.SIZE={self.cfg.DATA.SEARCH.SIZE}, "
                    f"effective_stride={self.effective_stride}"
                )
            self._heatmap_checked += 1

        assert gt_gaussian_maps.shape == pred_dict['score_map'].shape, (
            f"score_map shape mismatch: "
            f"pred={pred_dict['score_map'].shape}, "
            f"gt={gt_gaussian_maps.shape}. "
            f"arch_mode={self.arch_mode}, "
            f"effective_stride={self.effective_stride}"
        )

        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            nan_ratio = torch.isnan(pred_boxes).float().mean().item()
            print(f"[CEUTrackActor] WARNING: NaN in pred_boxes "
                  f"(ratio={nan_ratio:.3f}), skipping batch")
            dummy = torch.tensor(0.0, device=device, requires_grad=True)
            if return_status:
                return dummy, {
                    "Loss/total":    0.0,
                    "Loss/giou":     0.0,
                    "Loss/l1":       0.0,
                    "Loss/location": 0.0,
                    "IoU":           0.0,
                }
            return dummy

        num_queries    = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec   = (
            box_xywh_to_xyxy(gt_bbox)[:, None, :]
            .repeat(1, num_queries, 1)
            .view(-1, 4)
            .clamp(0.0, 1.0)
        )

        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except Exception as e:
            print(f"[CEUTrackActor] giou exception: {e}")
            giou_loss = torch.tensor(0.0, device=device)
            iou       = torch.tensor(0.0, device=device)

        l1_loss       = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        location_loss = (
            self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
            if 'score_map' in pred_dict
            else torch.tensor(0.0, device=device)
        )

        loss = (self.loss_weight['giou']  * giou_loss
              + self.loss_weight['l1']    * l1_loss
              + self.loss_weight['focal'] * location_loss)

        if return_status:
            mean_iou = iou.detach().mean() if iou.numel() > 1 else iou.detach()
            return loss, {
                "Loss/total"    : loss.item(),
                "Loss/giou"     : giou_loss.item(),
                "Loss/l1"       : l1_loss.item(),
                "Loss/location" : location_loss.item(),
                "IoU"           : mean_iou.item(),
            }
        return loss