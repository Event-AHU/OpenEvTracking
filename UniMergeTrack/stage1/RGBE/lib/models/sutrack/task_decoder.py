import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.utils.box_ops import box_xyxy_to_cxcywh

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_task_decoder(cfg, encoder):
    in_channel = encoder.num_channels
    hidden_dim = cfg.MODEL.DECODER.NUM_CHANNELS
    mlp_head = MLP(in_channel, hidden_dim, cfg.MODEL.TASK_NUM, 3)
    return mlp_head

