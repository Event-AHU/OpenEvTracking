import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

from flash_attn.layers.rotary import RotaryEmbedding
from lib.models.layers.rotary import apply_rotary_emb
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from .rms_norm import RMSNorm


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadFlashDiff2(nn.Module):
    """
    DiffAttn implemented with FlashAttention, for packages that does not support different qk/v dimensions
    e.g., flash-attention (https://github.com/Dao-AILab/flash-attention)
    """
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.device = torch.device("cuda")
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def make_filter_window(self, n_freq, cutoff_ratio=0.3, window_type='rect', device=None):
        if window_type == 'rect':
            cutoff = int(n_freq * cutoff_ratio)
            filt = torch.zeros(n_freq, device=device)
            filt[:cutoff] = 1.0
        elif window_type == 'hann':
            filt = torch.hann_window(n_freq, device=device)
        elif window_type == 'gaussian':
            x = torch.linspace(0, 1, n_freq, device=device)
            sigma = cutoff_ratio / 2.0  # 控制宽度
            filt = torch.exp(-0.5 * (x / sigma) ** 2)
        else:
            raise ValueError("window_type must be 'rect', 'hann', or 'gaussian'")
        return filt  # (n_freq,)

    def forward(
        self,
        query, 
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = query.size()

        rotary_emb = RotaryEmbedding(
            self.head_dim,
            base=10000.0,
            interleaved=True,
            device=self.device

        )
        rotary_emb._update_cos_sin_cache(tgt_len, device=self.device, dtype=torch.bfloat16)
        rel_pos = (rotary_emb._cos_cached, rotary_emb._sin_cached)
        
        src_len = tgt_len

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim).to(torch.bfloat16)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim).to(torch.bfloat16)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim).to(torch.bfloat16)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        attn11 = flash_attn_func(q1, k1, v1, causal=True)
        attn12 = flash_attn_func(q1, k1, v2, causal=True)
        attn1 = torch.cat([attn11, attn12], dim=-1)
        
        attn21 = flash_attn_func(q2, k2, v1, causal=True)
        attn22 = flash_attn_func(q2, k2, v2, causal=True)
        attn2 = torch.cat([attn21, attn22], dim=-1)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        attn1_freq = torch.fft.rfft(attn1.float(), dim=1)
        attn2_freq = torch.fft.rfft(attn2.float(), dim=1)

        attn_freq = attn1_freq - lambda_full * attn2_freq
        
        n_freq = attn_freq.shape[1]
        filt = self.make_filter_window(n_freq, window_type='gaussian', device=attn_freq.device)  # (n_freq,)
        filt = filt.view(1, n_freq, 1, 1)
        
        attn_freq = attn_freq * filt
                
        attn = torch.fft.irfft(attn_freq, n=attn1.shape[1], dim=1)
        
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        
        attn = self.out_proj(attn)
        return attn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AttentionLayer(nn.Module):
    def __init__(self, dim=768, num_heads=8, hidden_dim=768, drop_path=0.1, depth=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadFlashDiff2(dim, depth, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query):
        q = query + self.drop_path(self.attn(self.norm1(query)))
        q = q + self.drop_path(self.ffn(self.norm2(q)))
        return q


class AttentionTransformer(nn.Module):
    def __init__(self, dim=768, num_heads=8, hidden_dim=768, num_layers=6, drop_path=0.1):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.layers = nn.ModuleList([
            AttentionLayer(dim, num_heads, hidden_dim, dpr[i], depth=i)
            for i in range(num_layers)
        ])

    def forward(self, query):
        for layer in self.layers:
            query = layer(query)
        return query
