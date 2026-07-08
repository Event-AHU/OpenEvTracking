import math
import torch
import torch.nn as nn
from lib.utils.box_ops import generate_soft_mask
from lib.models.layers.hflayers import Hopfield


class FootprintRetrieval(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.hopfield = Hopfield(
            input_size=dim, hidden_size=dim, output_size=dim, pattern_size=dim,
            num_heads=num_heads, dropout=dropout, batch_first=True,
            normalize_state_pattern=True,
            normalize_state_pattern_affine=True,
            normalize_stored_pattern=True,
            normalize_stored_pattern_affine=True,
            normalize_pattern_projection=True,
            normalize_pattern_projection_affine=True,
            update_steps_max=0,
            update_steps_eps=1e-4,
        )
        self.fp_q_proj = nn.Linear(dim, dim, bias=False)
        self.fp_k_proj = nn.Linear(dim, dim, bias=False)
        fp_in_dim = num_heads * 2
        self.fp_weight = nn.Sequential(nn.Linear(fp_in_dim, max(num_heads * 4, 32)), nn.GELU(), nn.Linear(max(num_heads * 4, 32), dim), nn.Sigmoid())
        nn.init.zeros_(self.fp_weight[-2].weight)
        nn.init.constant_(self.fp_weight[-2].bias, 2.0)

    def forward(self, query, key, value, attn_bias=None, return_aux=False):
        key = self.key_proj(key)
        value = self.val_proj(value)
        if attn_bias is not None:
            B, N, M = attn_bias.shape
            attn_bias = attn_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, N, M)

        B, N, C = query.shape
        _, M, _ = key.shape
        head_dim = C // self.num_heads
        fp_q = self.fp_q_proj(query).reshape(B, N, self.num_heads, head_dim)
        fp_k = self.fp_k_proj(key).reshape(B, M, self.num_heads, head_dim) 
        fp_attn = torch.einsum('bnhd,bmhd->bnhm', fp_q, fp_k).mul_(head_dim ** -0.5).softmax(dim=-1)  
        fp_ent = -(fp_attn * fp_attn.clamp_min(1e-12).log()).sum(dim=-1) 
        fp_ent_norm = fp_ent / math.log(M)
        fp_maxprob = fp_attn.max(dim=-1).values
        fp_feat = torch.cat([fp_ent_norm, fp_maxprob], dim=-1)
        dim_weight = self.fp_weight(fp_feat)
        query = query * dim_weight
        fp_ent_norm_detach = fp_ent_norm.detach()
        fp_maxprob_detach = fp_maxprob.detach()
        dim_weight_detach = dim_weight.detach()
        fp_weight_channel = dim_weight_detach.mean(dim=1)
        fp_weight_channel_range = (fp_weight_channel.max(dim=-1).values - fp_weight_channel.min(dim=-1).values).mean()
        footprint_aux = {'fp_ent_norm_mean': fp_ent_norm_detach.mean(), 'fp_ent_norm_std': fp_ent_norm_detach.std(), 'fp_maxprob_mean': fp_maxprob_detach.mean(), 'fp_maxprob_std': fp_maxprob_detach.std(), 'fp_weight_mean': dim_weight_detach.mean(), 'fp_weight_std': dim_weight_detach.std(), 'fp_weight_channel_std': fp_weight_channel.std(), 'fp_weight_channel_range': fp_weight_channel_range}

        if return_aux:
            association_output, _, raw_attn, _ = self.hopfield._associate(data=(key, query, value), return_raw_associations=True, association_mask=attn_bias)
            out = self.hopfield._maybe_transpose(association_output)
            attn = raw_attn.detach().clamp_min(1e-12)
            entropy = -(attn * attn.log()).sum(dim=-1).mean(dim=(0, 2))
            maxprob = attn.max(dim=-1).values.mean(dim=(0, 2))
            aux = {'attn_entropy_mean': entropy.mean(), 'attn_maxprob_mean': maxprob.mean()}
            for head_idx in range(self.num_heads):
                aux[f'head{head_idx}_entropy'] = entropy[head_idx]
                aux[f'head{head_idx}_maxprob'] = maxprob[head_idx]
            aux.update(footprint_aux)
            return out, aux

        out = self.hopfield((key, query, value), association_mask=attn_bias)
        return out


class HopfieldCompletion(nn.Module):
    def __init__(self, dim=768, num_heads=4, dropout=0.1, size=256, stride=16, roi_bias_scale=1.0, memory_size=10, gate_init_value=-2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.size = size
        self.stride = stride
        self.roi_bias_scale = roi_bias_scale
        self.memory_size = memory_size
        self.rgb_retrieval = FootprintRetrieval(dim=dim, num_heads=num_heads, dropout=dropout)
        self.evt_retrieval = FootprintRetrieval(dim=dim, num_heads=num_heads, dropout=dropout)

        self.gate_i = nn.Sequential(nn.Linear(dim * 4, dim // 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim // 4, 1), nn.Sigmoid())
        self.gate_e = nn.Sequential(nn.Linear(dim * 4, dim // 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim // 4, 1), nn.Sigmoid())
        self._init_gate_weights(init_value=gate_init_value)

        self.reset_memory()

    def _init_gate_weights(self, init_value=0.0):
        for m in [self.gate_i, self.gate_e]:
            if isinstance(m[-2], nn.Linear):
                nn.init.constant_(m[-2].bias, init_value)

    def reset_memory(self):
        self.mem_rgb = []
        self.mem_evt = []
        self.mem_mask = []

    def store_memory(self, xi, xe, bbox):
        mask = generate_soft_mask(size=self.size, stride=self.stride, bbox=bbox).to(device=xi.device, dtype=xi.dtype)
        self.mem_rgb.append(xi.detach())
        self.mem_evt.append(xe.detach())
        self.mem_mask.append(mask.detach())
        if len(self.mem_rgb) > self.memory_size:
            self.mem_rgb.pop(1)
            self.mem_evt.pop(1)
            self.mem_mask.pop(1)

    def retrieve_memory(self, xi, xe, return_aux=False):
        if len(self.mem_rgb) == 0:
            if return_aux:
                aux_dict = {'xi_in': xi, 'xe_in': xe, 'xi_ret': torch.zeros_like(xi), 'xe_ret': torch.zeros_like(xe), 'xi_out': xi, 'xe_out': xe}
                return xi, xe, aux_dict
            return xi, xe

        mem_rgb = torch.cat(self.mem_rgb, dim=1)
        mem_evt = torch.cat(self.mem_evt, dim=1)
        mem_mask = torch.cat(self.mem_mask, dim=1)

        B, N, _ = xi.shape
        KN = mem_rgb.shape[1]
        roi_bias = self.roi_bias_scale * (mem_mask.clamp(min=0.0, max=1.0) - 1.0)
        roi_bias = roi_bias.unsqueeze(1).expand(B, N, KN)

        xi_in, xe_in = xi, xe
        if return_aux:
            xi_ret, rgb_aux = self.rgb_retrieval(query=xe, key=mem_evt, value=mem_rgb, attn_bias=roi_bias, return_aux=True)
        else:
            xi_ret = self.rgb_retrieval(query=xe, key=mem_evt, value=mem_rgb, attn_bias=roi_bias)
            rgb_aux = None
        delta_i = xi_ret - xi
        gate_i_in = torch.cat([xi, delta_i, torch.abs(delta_i), xi * delta_i], dim=-1)
        gate_i = self.gate_i(gate_i_in)
        gated_delta_i = gate_i * delta_i
        xi_out = xi + gated_delta_i

        if return_aux:
            xe_ret, evt_aux = self.evt_retrieval(query=xi, key=mem_rgb, value=mem_evt, attn_bias=roi_bias, return_aux=True)
        else:
            xe_ret = self.evt_retrieval(query=xi, key=mem_rgb, value=mem_evt, attn_bias=roi_bias)
            evt_aux = None
        delta_e = xe_ret - xe
        gate_e_in = torch.cat([xe, delta_e, torch.abs(delta_e), xe * delta_e], dim=-1)
        gate_e = self.gate_e(gate_e_in)
        gated_delta_e = gate_e * delta_e
        xe_out = xe + gated_delta_e

        if return_aux:
            aux_dict = {
                'xi_in': xi_in, 'xe_in': xe_in,
                'xi_ret': xi_ret, 'xe_ret': xe_ret,
                'xi_out': xi_out, 'xe_out': xe_out,
                'gate_i': gate_i, 'gate_e': gate_e,
                'delta_i_norm': delta_i.norm(dim=-1).mean().detach(),
                'delta_e_norm': delta_e.norm(dim=-1).mean().detach(),
            }
            if rgb_aux is not None and evt_aux is not None:
                aux_dict.update({
                    'gate_i_min': gate_i.detach().min(), 'gate_i_max': gate_i.detach().max(),
                    'gate_e_min': gate_e.detach().min(), 'gate_e_max': gate_e.detach().max(),
                    'gated_delta_i_norm': gated_delta_i.norm(dim=-1).mean().detach(),
                    'gated_delta_e_norm': gated_delta_e.norm(dim=-1).mean().detach(),
                    'xi_ret_norm': xi_ret.norm(dim=-1).mean().detach(),
                    'xe_ret_norm': xe_ret.norm(dim=-1).mean().detach(),
                })
                for key, val in rgb_aux.items():
                    aux_dict[f'rgb_{key}'] = val.detach()
                for key, val in evt_aux.items():
                    aux_dict[f'evt_{key}'] = val.detach()
                head_dim = self.dim // self.num_heads
                delta_i_head = delta_i.reshape(delta_i.shape[0], delta_i.shape[1], self.num_heads, head_dim).norm(dim=-1).mean(dim=(0, 1))
                delta_e_head = delta_e.reshape(delta_e.shape[0], delta_e.shape[1], self.num_heads, head_dim).norm(dim=-1).mean(dim=(0, 1))
                for head_idx in range(self.num_heads):
                    aux_dict[f'head{head_idx}_delta_i_norm'] = delta_i_head[head_idx].detach()
                    aux_dict[f'head{head_idx}_delta_e_norm'] = delta_e_head[head_idx].detach()
            return xi_out, xe_out, aux_dict
        return xi_out, xe_out
