"""
SSIU-FA (v2): Improved Structural Similarity-Inspired Unfolding Network
========================================================================
Architecture: SALK + SGA + Channel Attention with Global Residual Learning
Scale: x4 ONLY (hardcoded)
Channels: 64 | Blocks: 12
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Fixed Configuration ───────────────────────────────────────────────────────
SCALE = 4
EMBED_DIM = 64
NUM_BLOCKS = 28
# ────────────────────────────────────────────────────────────────────────────────


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
    Replaces standard MSGM with high-efficiency large receptive fields.
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
    Efficient for texture and periodic pattern restoration.
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        dtype = x.dtype
        global_feat = torch.mean(x, dim=(2, 3))
        spectral_gate = self.net(global_feat).view(B, C, 1, 1)

        # Force FFT to float32 to avoid ComplexHalf experimental support issues in AMP
        x_fft = torch.fft.rfft2(x.to(torch.float32), dim=(-2, -1), norm='ortho')
        x_fft = x_fft * spectral_gate.to(torch.float32)
        out = torch.fft.irfft2(x_fft, s=(H, W), dim=(-2, -1), norm='ortho')
        
        return out.to(dtype)


class ImprovedSSIUBlockV2(nn.Module):
    """Single block: SALK → SGA → Channel Attention with dual residual paths."""
    def __init__(self, dim):
        super().__init__()
        self.salk = SimilarityAwareLargeKernel(dim)
        self.sga = SpectralGateAttention(dim)
        self.ca = ChannelAttention(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-2, requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-2, requires_grad=True)

    def forward(self, x):
        identity = x

        # Path 1: SALK + SGA
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = identity + self.gamma1.view(1, -1, 1, 1) * self.sga(self.salk(x))

        # Path 2: Channel Attention refinement
        identity = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = identity + self.gamma2.view(1, -1, 1, 1) * self.ca(x)

        return x


class ImprovedSSIUNet(nn.Module):
    """
    SSIU-FA: Improved SSIU with SALK + SGA + Global Residual Learning.

    Key difference from prior versions:
    - Global residual: output = pixel_shuffle(features) + bicubic_upsample(input)
      This matches the baseline's architecture and is critical for convergence.
    """
    def __init__(self, upscale=SCALE, embed_dim=EMBED_DIM, num_blocks=NUM_BLOCKS):
        super().__init__()
        assert upscale == 4, f"This model is x4 ONLY. Got upscale={upscale}"

        self.upscale = upscale
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        self.conv_in = nn.Conv2d(3, embed_dim, 3, 1, 1)

        self.layers = nn.ModuleList([
            ImprovedSSIUBlockV2(embed_dim) for _ in range(num_blocks)
        ])

        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embed_dim, 3 * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        # Global residual: learn the RESIDUAL on top of bicubic upsampling
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear',
                             align_corners=False)

        f_in = self.conv_in(x)
        feat = f_in
        for layer in self.layers:
            feat = layer(feat)

        out = self.upsampler(self.conv_after(feat) + f_in)

        # Add global skip — critical for convergence
        return out + base

    @staticmethod
    def print_config():
        """Print the fixed model configuration for verification."""
        model = ImprovedSSIUNet()
        params_k = sum(p.numel() for p in model.parameters()) / 1e3
        print("=" * 55)
        print("  SSIU-FA (Improved) — Architecture Configuration")
        print("=" * 55)
        print(f"  Scale factor  : x{SCALE}")
        print(f"  Embed dim     : {EMBED_DIM}")
        print(f"  Num blocks    : {NUM_BLOCKS}")
        print(f"  Parameters    : {params_k:.1f} K")
        print(f"  Global residual: YES (bicubic skip)")
        print("=" * 55)
        return model
