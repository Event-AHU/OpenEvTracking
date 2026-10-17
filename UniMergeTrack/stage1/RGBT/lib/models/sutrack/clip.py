import torch
import clip
import torch.nn as nn
from timm.models.layers import trunc_normal_

class TextEncoder(nn.Module):
    def __init__(self, type, out_channel):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, self.preprocess = clip.load(type, device=device)
        clip_embed_dim = self.clip.text_projection.size(1)
        self.norm1 = nn.LayerNorm(clip_embed_dim)
        self.text_proj = nn.Linear(clip_embed_dim, out_channel)
        self.norm2 = nn.LayerNorm(out_channel)
        trunc_normal_(self.text_proj.weight, std=.02)
        nn.init.constant_(self.text_proj.bias, 0)
    @property
    def dtype(self):
        return self.text_proj.weight.dtype
    def forward(self, text_data):
        text_src = self.clip.encode_text(text_data).type(self.dtype)
        # text_src = self.norm1(text_src)
        text_src = self.text_proj(text_src)
        # text_src = self.norm2(text_src)
        text_src = text_src.unsqueeze(1)
        return text_src

def build_textencoder(cfg, encoder):
    num_channels_enc = encoder.num_channels
    model = TextEncoder(cfg.MODEL.TEXT_ENCODER.TYPE, num_channels_enc)
    return model