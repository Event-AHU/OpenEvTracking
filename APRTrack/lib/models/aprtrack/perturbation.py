import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.box_ops import generate_soft_mask


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)


class AdversarialModalityGate(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 4, 32)
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def forward(self, xi, xe, alpha=1.0):
        pooled = torch.cat([xi.mean(dim=1), xe.mean(dim=1)], dim=-1)
        logits = self.net(self.grl(pooled, alpha))
        probs = torch.softmax(logits, dim=-1)
        choice = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        rgb_keep = 1.0 - choice[:, 1:2]
        evt_keep = 1.0 - choice[:, 2:3]
        return rgb_keep.view(-1, 1, 1), evt_keep.view(-1, 1, 1), probs


class AnchorLocalSpatialMask(nn.Module):
    def __init__(self, dim, mask_ratio_min=0.2, mask_ratio_max=0.4, temperature=1.0, size=256, stride=16, anchor_overlap_threshold=0.45, anchor_penalty_weight=0.5):
        super().__init__()
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.temperature = temperature
        self.size = size
        self.stride = stride
        self.anchor_overlap_threshold = anchor_overlap_threshold
        self.anchor_penalty_weight = anchor_penalty_weight
        self.grl = GradientReversalLayer()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self._window_cache = {}

    def _candidate_windows(self, h, w, device, dtype):
        cache_key = (h, w, self.mask_ratio_min, self.mask_ratio_max)
        if cache_key not in self._window_cache:
            total = h * w
            min_area = max(1, int(round(total * self.mask_ratio_min)))
            max_area = max(min_area, int(round(total * self.mask_ratio_max)))
            windows = []
            for win_h in range(1, h + 1):
                for win_w in range(1, w + 1):
                    area = win_h * win_w
                    if min_area <= area <= max_area:
                        for top in range(0, h - win_h + 1):
                            for left in range(0, w - win_w + 1):
                                mask = torch.zeros(h, w, dtype=torch.float32)
                                mask[top:top + win_h, left:left + win_w] = 1.0
                                windows.append(mask.reshape(-1))
            if not windows:
                side = max(1, int(round(math.sqrt(total * self.mask_ratio_min))))
                side = min(side, h, w)
                mask = torch.zeros(h, w, dtype=torch.float32)
                mask[:side, :side] = 1.0
                windows.append(mask.reshape(-1))
            self._window_cache[cache_key] = torch.stack(windows, dim=0)
        return self._window_cache[cache_key].to(device=device, dtype=dtype)

    def forward(self, x, hw_shape, alpha=1.0, bbox=None):
        if bbox is None:
            raise ValueError('Spatial perturbation requires bbox')
        h, w = hw_shape
        token_logits = self.score(self.grl(x, alpha)).squeeze(-1)
        windows = self._candidate_windows(h, w, x.device, x.dtype)
        target_mask = generate_soft_mask(size=self.size, stride=self.stride, bbox=bbox).to(device=x.device, dtype=x.dtype)
        target_area = target_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        target_overlap = torch.matmul(target_mask, windows.transpose(0, 1)) / target_area
        anchor_penalty = F.relu(target_overlap - self.anchor_overlap_threshold)
        base_scores = torch.matmul(token_logits, windows.transpose(0, 1))
        window_scores = base_scores - self.anchor_penalty_weight * anchor_penalty
        window_weight = F.gumbel_softmax(window_scores, tau=self.temperature, hard=True, dim=-1)
        rect_mask = torch.matmul(window_weight, windows)
        keep_mask_hard = 1.0 - rect_mask
        keep_mask_soft = 1.0 - torch.matmul(torch.softmax(window_scores, dim=-1), windows)
        keep_mask = keep_mask_hard.detach() - keep_mask_soft.detach() + keep_mask_soft
        stats = {
            'apg/spatial_target_overlap': (window_weight * target_overlap).sum(dim=-1).mean().detach(),
            'apg/spatial_anchor_penalty': (window_weight * anchor_penalty).sum(dim=-1).mean().detach()
        }
        return keep_mask.unsqueeze(-1), keep_mask_soft.unsqueeze(-1), stats


class AdversarialPerturbationModule(nn.Module):
    def __init__(self, dim, enable_modal=True, enable_spatial=True, spatial_mask_ratio_min=0.2, spatial_mask_ratio_max=0.4, spatial_temperature=1.0, spatial_anchor_overlap_threshold=0.45, spatial_anchor_penalty_weight=0.5, size=256, stride=16, modal_balance_weight=0.1, modal_target_probs=None, route_probs=None, route_completion=None):
        super().__init__()
        self.enable_modal = enable_modal
        self.enable_spatial = enable_spatial
        self.modal_balance_weight = modal_balance_weight
        if modal_target_probs is None:
            modal_target_probs = [0.2, 0.4, 0.4]
        if route_probs is None:
            route_probs = [0.4, 0.3, 0.3]
        if route_completion is None:
            route_completion = [True, False, True]
        route_probs = torch.tensor(route_probs, dtype=torch.float32)
        if route_probs.numel() != 3 or route_probs.sum() <= 0:
            raise ValueError('route_probs must be a length-3 list with a positive sum')
        route_completion = torch.tensor(route_completion, dtype=torch.bool)
        if route_completion.numel() != 3:
            raise ValueError('route_completion must be a length-3 list')
        self.register_buffer('route_probs', route_probs / route_probs.sum())
        self.register_buffer('route_completion', route_completion)
        self.register_buffer('modal_target_probs', torch.tensor(modal_target_probs, dtype=torch.float32))
        self.modal_gate = AdversarialModalityGate(dim) if enable_modal else None
        if enable_spatial:
            spatial_kwargs = {'dim': dim, 'mask_ratio_min': spatial_mask_ratio_min, 'mask_ratio_max': spatial_mask_ratio_max, 'temperature': spatial_temperature, 'size': size, 'stride': stride, 'anchor_overlap_threshold': spatial_anchor_overlap_threshold, 'anchor_penalty_weight': spatial_anchor_penalty_weight}
            self.spatial_rgb = AnchorLocalSpatialMask(**spatial_kwargs)
            self.spatial_evt = AnchorLocalSpatialMask(**spatial_kwargs)
        else:
            self.spatial_rgb = None
            self.spatial_evt = None

    def _apply_spatial(self, xi, xe, alpha, hw_shape, bbox=None):
        if hw_shape is None:
            raise ValueError('Spatial rectangle mask requires hw_shape')
        spatial_rgb = self.spatial_rgb
        spatial_evt = self.spatial_evt
        rgb_mask, rgb_soft, rgb_stats = spatial_rgb(xi, hw_shape, alpha, bbox=bbox)
        evt_mask, evt_soft, evt_stats = spatial_evt(xe, hw_shape, alpha, bbox=bbox)
        stats = {'apg/spatial_rgb_keep': rgb_soft.mean().detach(), 'apg/spatial_evt_keep': evt_soft.mean().detach()}
        for key in rgb_stats:
            stats[key] = rgb_stats[key]
        for key in evt_stats:
            evt_key = key.replace('apg/', 'apg/evt_') if key.startswith('apg/') else f'evt_{key}'
            stats[evt_key] = evt_stats[key]
        return xi * rgb_mask, xe * evt_mask, stats

    def _apply_modal(self, xi, xe, alpha, modal_scale):
        rgb_keep, evt_keep, probs = self.modal_gate(xi, xe, alpha)
        rgb_keep = 1.0 - (1.0 - rgb_keep) * modal_scale
        evt_keep = 1.0 - (1.0 - evt_keep) * modal_scale
        mean_probs = probs.mean(dim=0)
        modal_reg_loss = F.kl_div(torch.log(mean_probs.clamp_min(1e-6)), self.modal_target_probs, reduction='batchmean')
        stats = {
            'apg/modal_none_prob': probs[:, 0].mean().detach(),
            'apg/modal_rgb_missing_prob': probs[:, 1].mean().detach(),
            'apg/modal_evt_missing_prob': probs[:, 2].mean().detach(),
            'apg/modal_reg_loss': modal_reg_loss.detach(),
            'apg/modal_scale': modal_scale
        }
        return xi * rgb_keep, xe * evt_keep, stats, modal_reg_loss

    def forward(self, xi, xe, alpha=1.0, hw_shape=None, bbox=None, modal_scale=1.0):
        stats = {}
        aux_loss = xi.new_zeros(())
        enable_modal = self.enable_modal
        enable_spatial = self.enable_spatial
        route = torch.multinomial(self.route_probs, xi.shape[0], replacement=True)
        if not enable_modal and (route == 1).any():
            route[route == 1] = 0
        if not enable_spatial and (route == 2).any():
            route[route == 2] = 0
        none_mask = route == 0
        modal_mask = route == 1
        spatial_mask = route == 2
        xi_out = xi.clone()
        xe_out = xe.clone()

        stats['apg/route_none_ratio'] = none_mask.float().mean().detach()
        stats['apg/route_modal_ratio'] = modal_mask.float().mean().detach()
        stats['apg/route_spatial_ratio'] = spatial_mask.float().mean().detach()
        stats['apg/route_completion_ratio'] = self.route_completion[route].float().mean().detach()

        if modal_mask.any():
            if not enable_modal:
                raise ValueError('modal_enable must be enabled when route samples layer1')
            xi_modal, xe_modal, modal_stats, modal_reg_loss = self._apply_modal(xi[modal_mask], xe[modal_mask], alpha, modal_scale)
            xi_out[modal_mask] = xi_modal
            xe_out[modal_mask] = xe_modal
            aux_loss = aux_loss + modal_scale * self.modal_balance_weight * modal_reg_loss
            stats.update(modal_stats)

        if spatial_mask.any():
            if not enable_spatial:
                raise ValueError('spatial_enable must be enabled when route samples layer2')
            spatial_bbox = bbox[spatial_mask] if bbox is not None else None
            xi_spatial, xe_spatial, spatial_stats = self._apply_spatial(xi[spatial_mask], xe[spatial_mask], alpha, hw_shape, bbox=spatial_bbox)
            xi_out[spatial_mask] = xi_spatial
            xe_out[spatial_mask] = xe_spatial
            stats.update(spatial_stats)

        return xi_out, xe_out, stats, aux_loss, self.route_completion[route]
