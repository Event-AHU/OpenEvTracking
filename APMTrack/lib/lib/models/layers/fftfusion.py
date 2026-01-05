import torch
from torch import nn
import torch.nn.functional as F

class FFTFusion(nn.Module):
    def __init__(self, sigma=7, channels=3):
        super().__init__()
        self.sigma = sigma
        
        # Frequency domain preprocessing
        self.freq_preprocess = nn.Conv2d(channels, 1, kernel_size=1)
        self.freq_preprocess_event = nn.Conv2d(5, 1, kernel_size=1)
        
        # Amplitude and phase processing blocks
        self.process_amp = self._make_process_block(1)
        self.process_pha = self._make_process_block(1)
        self.process_amp_event = self._make_process_block(1)
        self.process_pha_event = self._make_process_block(1)
        
        self.conv1_fft = nn.Conv2d(2, 2, kernel_size=1)
        self.conv2_fft = nn.Conv2d(2, 2, kernel_size=1)
        
        # self.freq_preprocess_back = nn.Conv2d(1, 1, kernel_size=1)

    def _make_process_block(self, channels=1):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def make_gaussian(self, height, width):
        """Create centered Gaussian filter for half-spectrum (rfft)"""
        y_center, x_center = height // 2, 0
        
        y_coords = torch.arange(height).float() - y_center
        x_coords = torch.arange(width // 2 + 1).float() - x_center
        
        dist_sq = y_coords[:, None]**2 + x_coords[None, :]**2
        g = torch.exp(-dist_sq / (2 * self.sigma**2))
        
        return g.unsqueeze(0).unsqueeze(0)  # Shape: [1, 3, H, W//2+1]

    def multiply_and_softmax(self, vis, event):
        # Normalize features to avoid numerical instability
        vis = F.normalize(vis, dim=1)
        event = F.normalize(event, dim=1)

        # Flatten and multiply
        features1_flattened = vis.view(vis.size(0), vis.size(1), -1)
        features2_flattened = event.view(event.size(0), event.size(1), -1)
        multiplied = torch.mul(features1_flattened, features2_flattened)

        # Apply softmax
        multiplied_softmax = torch.softmax(multiplied, dim=2)
        multiplied_softmax = multiplied_softmax.view(vis.size(0), vis.size(1), vis.size(2), vis.size(3))

        # Residual connection
        vis_map = vis * multiplied_softmax + vis
        return vis_map
    
    def forward(self, x, event_x):
        B, C, H, W = x.shape
        
        # FFT Transform
        x_freq = torch.fft.rfft2(self.freq_preprocess(x), norm='backward')
        event_freq = torch.fft.rfft2(self.freq_preprocess_event(event_x), norm='backward')
        
        # Frequency separation
        gaussian_filter = self.make_gaussian(H, W).to(x.device)

        # Extract high freq from events (typically contains edge info)
        event_high_freq = event_freq * (1 - gaussian_filter)
        
        # Get amplitude and phase
        x_amp, x_pha = torch.abs(x_freq), torch.angle(x_freq)
        event_amp, event_pha = torch.abs(event_high_freq), torch.angle(event_high_freq)
        
        # Process components
        x_amp = self.process_amp(x_amp)
        x_pha = self.process_pha(x_pha)
        event_amp = self.process_amp_event(event_amp)
        event_pha = self.process_pha_event(event_pha)
        
        # Adaptive fusion
        # fused_amp = x_amp * torch.sigmoid(event_amp) + x_amp
        fused_amp = self.multiply_and_softmax(x_amp, event_amp)
        fused_pha = self.multiply_and_softmax(x_pha, event_pha)
        
        # Reconstruct complex tensor
        real = fused_amp * torch.cos(fused_pha)
        imag = fused_amp * torch.sin(fused_pha)
        
        f = torch.cat((real, imag), dim=1)
        f = F.relu(self.conv1_fft(f))
        f = self.conv2_fft(f).float()
        
        real, imag = torch.chunk(f, 2, dim=1)
        fused_freq = torch.complex(real, imag)
        
        # Inverse FFT with normalization
        x_out = torch.fft.irfft2(fused_freq, s=(H, W), norm='backward')
        
        x = x + x_out
        
        return x