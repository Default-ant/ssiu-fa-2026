import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, max(dim // reduction, 8), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(dim // reduction, 8), dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        return x * self.fc(y)

class SimilarityAwareLargeKernel(nn.Module):
    """
    SALK: Captures long-range structural similarity using decomposed 13x13 kernels.
    Replaces standard MSGM/spectral gating with high-efficiency large receptive fields.
    """
    def __init__(self, dim):
        super().__init__()
        # Decomposed 13x13 kernel: 1x13 then 13x1
        self.lk1 = nn.Conv2d(dim, dim, (1, 13), padding=(0, 6), groups=dim)
        self.lk2 = nn.Conv2d(dim, dim, (13, 1), padding=(6, 0), groups=dim)
        # Refinement 3x3 for local continuity
        self.lk3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
        self.proj = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.lk1(x)
        out = self.lk2(out)
        out = out + self.lk3(x)
        return self.proj(out * self.gate(x))

class SpectralGateAttention(nn.Module):
    """
    SGA: Global attention in the Frequency Domain via FFT. 
    Extremely efficient for texture and periodic pattern restoration.
    """
    def __init__(self, dim):
        super().__init__()
        # Simplified Spectral Gate: Learns frequency coefficients
        # Using a small MLP to generate weights instead of direct weights to save space/be dynamic
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Global Frequency Features via global average pool
        global_feat = torch.mean(x, dim=(2, 3))
        spectral_gate = self.net(global_feat).view(B, C, 1, 1)
        
        # Fourier Transform (Real-to-Complex)
        # Using fft.rfft2 for efficiency on real-valued images
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # Mode-wise Modulation
        x_fft = x_fft * spectral_gate
        
        # Inverse Fourier Transform
        out = torch.fft.irfft2(x_fft, s=(H, W), dim=(-2, -1), norm='ortho')
        return out

class ImprovedSSIUBlockV2(nn.Module):
    """
    Combining SALK and SGA for a total structural similarity and frequency recovery.
    """
    def __init__(self, dim):
        super().__init__()
        self.salk = SimilarityAwareLargeKernel(dim)
        self.sga = SpectralGateAttention(dim)
        self.ca = ChannelAttention(dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Scale parameters for training stability (SOTA standard)
        self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-2, requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-2, requires_grad=True)

    def forward(self, x):
        # LayerNorm expects [B, H, W, C] but we have [B, C, H, W]
        identity = x
        
        # Local-Global SALK path
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = identity + self.gamma1.view(1, -1, 1, 1) * self.sga(self.salk(x))
        
        # Channel-Frequency Refinement
        identity = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = identity + self.gamma2.view(1, -1, 1, 1) * self.ca(x)
        
        return x

class ImprovedSSIUNet(nn.Module):
    """
    SSIU-V2: Unified Dual-Domain Unfolding network.
    Designed to beat the 32.64 dB SSIU (2025) baseline.
    """
    def __init__(self, upscale=4, embed_dim=64, num_blocks=12):
        super().__init__()
        self.conv_in = nn.Conv2d(3, embed_dim, 3, 1, 1)
        
        self.layers = nn.ModuleList([
            ImprovedSSIUBlockV2(embed_dim) for _ in range(num_blocks)
        ])
        
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embed_dim, 3 * (upscale**2), 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        f_in = self.conv_in(x)
        feat = f_in
        for layer in self.layers:
            feat = layer(feat)
        
        out = self.upsampler(self.conv_after(feat) + f_in)
        return out
