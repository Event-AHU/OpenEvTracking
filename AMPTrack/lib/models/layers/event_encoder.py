import torch.nn as nn


class EventEncoder(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),        # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),       # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),      # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        """
        输入: x (B, T, H, W)
        输出: (B, T, h*w, 768)
        """
        B, T, H, W = x.shape  # (64, 5, 256, 256)
        x = x.view(B * T, 1, H, W)    # [240, 1, 256, 256]
        
        feats = self.encoder(x)       # [240, 256, 16, 16]
        _, _, h, w = feats.shape          
    
        feats = feats.permute(0, 2, 3, 1).contiguous().reshape(B, T, h*w, -1)  # (B, T, 256, C)
        out_feats = self.proj(feats).reshape(B, T, h, w, -1) #  [48, 5, 16, 16, 768]
        return  out_feats