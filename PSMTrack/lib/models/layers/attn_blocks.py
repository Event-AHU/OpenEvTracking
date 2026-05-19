import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention
import torch.nn.functional as F
from functools import partial
import collections.abc
from itertools import repeat


class PolicyNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, out_dim)
        )

    def forward(self, x, temp, hard=True, num_active=3):
        # x shape: [B, in_dim]
        logits = self.net(x)
        active_logits = logits[:, :num_active] 
        mask = F.gumbel_softmax(active_logits, tau=temp, hard=hard, dim=-1)

        return mask

    
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            cur_layer=None,
            use_conv=False,
            expert_layer=[0, 6, 10]
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        # for param in self.fc1.parameters(): param.requires_grad = False
        # for param in self.fc2.parameters(): param.requires_grad = False
        # if norm_layer is not None:
        #     for param in self.norm.parameters(): param.requires_grad = False
    
        self.router = PolicyNet(in_dim=in_features * 2, out_dim=3)
        
        self.k = 3  
        split_hidden = hidden_features // self.k  # H/k

        self.fc1_experts = nn.ModuleList([
            linear_layer(in_features, split_hidden, bias=bias[0])
            for _ in range(self.k)
        ])

        self.fc2_experts = nn.ModuleList([
            linear_layer(split_hidden, out_features, bias=bias[1])
            for _ in range(self.k)
        ])
                
        if hasattr(self, 'fc1'):
            with torch.no_grad():
                W = self.fc1.weight.data         # [3072, 768]
                b = self.fc1.bias.data if self.fc1.bias is not None else None  # [3072]
                for i, fc in enumerate(self.fc1_experts):
                    start = i * split_hidden
                    end = (i+1) * split_hidden
                    fc.weight.data.copy_(W[start:end, :])  
                    if b is not None:
                        fc.bias.data.copy_(b[start:end])

        if hasattr(self, 'fc2'):
            with torch.no_grad():
                W2 = self.fc2.weight.data      # [768, 3072]
                b2 = self.fc2.bias.data
                for i, fc in enumerate(self.fc2_experts):
                    start = i * split_hidden
                    end = (i+1) * split_hidden
                    fc.weight.data.copy_(W2[:, start:end])  
                    if b2 is not None:
                        fc.bias.data.copy_(b2 / self.k)             
        
        self.expert1 = nn.Sequential(self.fc1_experts[0], 
                                     self.act,
                                     self.drop1,
                                     self.fc2_experts[0],
                                     self.drop2)
        self.expert2 = nn.Sequential(self.fc1_experts[1], 
                                     self.act,
                                     self.drop1,
                                     self.fc2_experts[1],
                                     self.drop2)
        self.expert3 = nn.Sequential(self.fc1_experts[2], 
                                     self.act,
                                     self.drop1,
                                     self.fc2_experts[2],
                                     self.drop2)
        
        if cur_layer not in expert_layer:
            for param in self.router.parameters(): param.requires_grad = False
            
            for param in self.expert1.parameters(): param.requires_grad = False
            for param in self.expert2.parameters(): param.requires_grad = False
            for param in self.expert3.parameters(): param.requires_grad = False
            
        self.expert_layer = expert_layer
        
    def router_forward(self, x, chunk_lens):
        B, N, C = x.shape
        TEMPLATE_LEN = 64
        
        num_active = len(chunk_lens) 
        experts = [self.expert1, self.expert2, self.expert3]

        template = x[:, :TEMPLATE_LEN]
        search = x[:, TEMPLATE_LEN:]
        search_chunks = torch.split(search, chunk_lens, dim=1)
        
        template_feat = template.mean(1)
        search_feats = torch.stack([chunk.mean(1) for chunk in search_chunks], dim=1) # [B, num_active, C]
        search_context = search_feats.mean(dim=1) # [B, C] 
        router_inp = torch.cat([template_feat, search_context], dim=-1) 
        
        policy_out = self.router(router_inp, temp=1, num_active=num_active) 
         
        if self.training:
            cur_feats = []
            for i in range(num_active):
                inp = torch.cat((template, search_chunks[i]), dim=1)
                cur_feats.append(experts[i](inp))
            feat_stack = torch.stack(cur_feats, dim=1)
            x_route = torch.einsum("brnc,br -> bnc", feat_stack, policy_out) if policy_out is not None else  feat_stack.squeeze(1)   
        else:
            expert_idx = policy_out.argmax(dim=1).item() if policy_out is not None else 0
            selected_expert = experts[expert_idx]
            search_in = search_chunks[expert_idx]
            x_route = selected_expert(torch.cat([template, search_in], dim=1))

        template = x_route[:,:TEMPLATE_LEN]
        route_search = x_route[:,TEMPLATE_LEN:]
        route_search = route_search.repeat(1,num_active,1)
        x_route = torch.cat((template, route_search), dim=1)
        
        return x_route
        
        
    def forward(self, x, chunk_lens, layer):
        if layer in self.expert_layer:
            x_route = self.router_forward(x, chunk_lens)
        else:
            x_route = None
                    
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        if x_route is not None:
            x = x + x_route 
 
        return x

class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cur_layer=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, cur_layer=cur_layer)

    def forward(self, x, mask=None, return_attention=False, chunk_lens=None, layer=None):
        if return_attention:
            attn_out, attn_map = self.attn(self.norm1(x), mask, return_attention=True)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x), chunk_lens=chunk_lens, laye=layer))
            return x, attn_map
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), mask))
            x = x + self.drop_path(self.mlp(self.norm2(x), chunk_lens=chunk_lens, layer=layer))
            return x