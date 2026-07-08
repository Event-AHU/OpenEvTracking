from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.box_ops import generate_soft_mask


class APRTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg


    def __call__(self, data):
        out_dict_list = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict_list)
        return loss, status


    def forward_pass(self, data):
        epoch = data.get('epoch', None)
        z = data['template_images']
        x = data['search_images']
        zi, ze = z[:, :, :3, :, :], z[:, :, 3:, :, :]
        xi, xe = x[:, :, :3, :, :], x[:, :, 3:, :, :]
        zb = data['template_anno']
        xb = data['search_anno']

        mask_z = []
        for i in range(self.settings.num_template):
            mask_z.append(generate_soft_mask(size=self.cfg.DATA.TEMPLATE.SIZE, stride=self.cfg.MODEL.BACKBONE.STRIDE, bbox=zb[:, i]))
        mask_z = torch.cat(mask_z, dim=1)

        B, M, C, H, W = zi.size()
        zi = zi.reshape(-1, C, H, W)
        ze = ze.reshape(-1, C, H, W)
        zi, ze = self.net.backbone.patch_embed(zi, ze)
        for blk in self.net.backbone.blocks[:-self.net.backbone.num_main_blocks]:
            zi, ze = blk(zi, ze)
        zi = zi.flatten(2)
        ze = ze.flatten(2)
        self.net.apg_completion_mask = None
        zi += self.net.backbone.pos_embed_z
        ze += self.net.backbone.pos_embed_z
        mask = mask_z.reshape(-1, self.net.zs, 1)
        z_pos_embed = self.net.backbone.zf_pos_embed * mask + self.net.backbone.zb_pos_embed * (1 - mask)
        zi += z_pos_embed
        ze += z_pos_embed
        zi = zi.reshape(B, -1, self.net.embed_dim)
        ze = ze.reshape(B, -1, self.net.embed_dim)
        self.net.apg_stats = {}
        self.net.apg_aux_loss = zi.new_zeros(())

        M = xi.size(1)
        xis, xes = [], []
        for idx in range(M):
            xi_i = xi[:, idx:idx+1, :, :, :]
            xe_i = xe[:, idx:idx+1, :, :, :]
            B, M_i, C, H, W = xi_i.size()
            xi_i = xi_i.reshape(-1, C, H, W)
            xe_i = xe_i.reshape(-1, C, H, W)
            grad_ctx = torch.no_grad() if idx < M - 1 else torch.enable_grad()
            with grad_ctx:
                xi_i, xe_i = self.net.backbone.patch_embed(xi_i, xe_i)
                for blk in self.net.backbone.blocks[:-self.net.backbone.num_main_blocks]:
                    xi_i, xe_i = blk(xi_i, xe_i)
                xi_i = xi_i.flatten(2)
                xe_i = xe_i.flatten(2)
                apg_stats = {}
                apg_aux_loss = xi_i.new_zeros(())
                self.net.apg_completion_mask = None
                if idx == M - 1 and self.net.use_apg and self.net.training:
                    if epoch is None:
                        alpha, modal_scale = 0.0, 0.0
                    elif self.net.apg_warmup_epochs <= 0:
                        alpha, modal_scale = 1.0, 1.0
                    else:
                        scale = min(1.0, float(epoch) / float(self.net.apg_warmup_epochs))
                        alpha, modal_scale = scale, scale
                    hw_tokens = int(round(xi_i.shape[1] ** 0.5))
                    xi_i, xe_i, apg_stats, apg_aux_loss, apg_completion_mask = self.net.apg(xi_i, xe_i, alpha=alpha, hw_shape=(hw_tokens, hw_tokens), bbox=xb[:, idx, :].reshape(-1, 4), modal_scale=modal_scale)
                    self.net.apg_completion_mask = apg_completion_mask
                xi_i += self.net.backbone.pos_embed_x
                xe_i += self.net.backbone.pos_embed_x
                xi_i += self.net.backbone.x_pos_embed
                xe_i += self.net.backbone.x_pos_embed
                xi_i = xi_i.reshape(B, -1, self.net.embed_dim)
                xe_i = xe_i.reshape(B, -1, self.net.embed_dim)
                self.net.apg_stats = apg_stats
                self.net.apg_aux_loss = apg_aux_loss
            xis.append(xi_i)
            xes.append(xe_i)

        self.net.reset_memory()
        out_dict_list = []
        for idx, (xi_i, xe_i) in enumerate(zip(xis, xes)):
            bbox = xb[:, idx, :]
            completion_aux = None
            if self.net.completion is not None and idx < len(xis) - 1:
                self.net.completion.store_memory(xi_i, xe_i, bbox)
                continue

            out_dict, attn_xi, attn_xe, completion_aux = self.net(zi, ze, xi_i, xe_i, None, return_completion_aux=True, epoch=epoch)
            self.net.temporal_stats = {}
            out_dict['bbox'] = bbox
            out_dict['apg_aux_loss'] = self.net.apg_aux_loss
            out_dict.update(self.net.apg_stats)
            out_dict.update(self.net.temporal_stats)
            if completion_aux is not None:
                if 'gate_i' in completion_aux and 'gate_e' in completion_aux:
                    out_dict.update({'completion/xi_gate': completion_aux['gate_i'].detach().mean(), 'completion/xe_gate': completion_aux['gate_e'].detach().mean()})
                for k, v in completion_aux.items():
                    if k in ('gate_i', 'gate_e', 'xi_in', 'xe_in', 'xi_ret', 'xe_ret', 'xi_out', 'xe_out'):
                        continue
                    out_dict[f'completion/{k}'] = v if not torch.is_tensor(v) else v.detach()
            out_dict_list.append(out_dict)
        return out_dict_list


    def compute_losses(self, out_dict_list, return_status=True):
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()
        for i, out_dict in enumerate(out_dict_list):
            gt_bbox = out_dict['bbox']
            gt_gaussian_map = generate_heatmap(gt_bbox.unsqueeze(1).permute(1, 0, 2), self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)[0].unsqueeze(1)
            pred_boxes = out_dict['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).view(-1, 4).clamp(min=0.0, max=1.0)
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
            if 'score_map' in out_dict:
                location_loss = self.objective['focal'](out_dict['score_map'], gt_gaussian_map)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
            apg_aux_loss = out_dict.get('apg_aux_loss', torch.tensor(0.0, device=loss.device))
            loss = loss + apg_aux_loss
            total_loss += loss
            if return_status:
                mean_iou = iou.detach().mean()
                status = {f'{i}frame_Loss/total': loss.item(), f'{i}frame_Loss/giou': giou_loss.item(), f'{i}frame_Loss/l1': l1_loss.item(), f'{i}frame_Loss/location': location_loss.item(), f'{i}frame_Loss/apg_aux': apg_aux_loss.item(), f'{i}frame_IoU': mean_iou.item()}
                total_status.update(status)
                for key, val in out_dict.items():
                    if key.startswith('apg/') or key.startswith('completion/'):
                        total_status[key] = float(val.item() if torch.is_tensor(val) else val)
        return total_loss, total_status if return_status else total_loss
