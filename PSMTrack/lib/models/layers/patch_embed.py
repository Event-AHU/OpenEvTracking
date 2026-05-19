import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        
        # self.proj_small = nn.Sequential(nn.Conv2d(in_chans, 384, kernel_size=8, stride=8),
        #                                 nn.Conv2d(384, embed_dim, kernel_size=2, stride=2)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.proj_large = nn.Conv2d(in_chans, embed_dim, kernel_size=32, stride=patch_size, padding=8)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, multi_search=False):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
    
        if multi_search:    
            x_sparse = self.proj(x[0])
            # x_sparse = F.avg_pool2d(x_sparse, kernel_size=2, stride=2)
            
            x_mid = self.proj(x[1])
            # x_mid = F.avg_pool2d(x_mid, kernel_size=2, stride=2)
            
            x_dense = self.proj(x[2])
            # x_dense = F.avg_pool2d(x_dense, kernel_size=2, stride=2)
   
            if self.flatten:
                ## x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
                x_sparse = x_sparse.flatten(2).transpose(1, 2)  # [B, N, C]
                x_mid = x_mid.flatten(2).transpose(1, 2)
                x_dense = x_dense.flatten(2).transpose(1, 2)

            x_sparse = self.norm(x_sparse)
            x_mid = self.norm(x_mid)
            x_dense = self.norm(x_dense)

            # x = self.norm(x)

            return x_sparse, x_mid, x_dense  

        else:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = self.norm(x)
                
        return x
