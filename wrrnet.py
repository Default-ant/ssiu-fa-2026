"""
WRRNet: Wavelet-Routed Reparameterization Network for Image Super-Resolution
=============================================================================
Architecture designed to beat SSIU (arXiv:2506.11823) on Set5/Set14/etc.

Key ideas:
  - Spatial sparsity: Haar DWT finds edges; only those pixels go through
    the heavy RepSR branch. Flat backgrounds take a cheap 1x1 path.
  - Structural re-parameterization: multi-branch training collapses to a
    single 3x3 conv at inference, keeping latency minimal.
  - Global skip: bilinear upsample of LR added to final output (like EDSR).

Usage (training):
    model = WRRNet(scale=4, num_channels=64, num_blocks=8)
    out   = model(lr_image)          # forward with Gumbel-Softmax routing

Usage (inference / after training):
    model.reparameterize()           # fuse multi-branch weights once
    out = model(lr_image)            # now uses fused single-conv path
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. HaarDWTEdgePrior
# ---------------------------------------------------------------------------

class HaarDWTEdgePrior(nn.Module):
    """
    Fixed (no-gradient) Haar wavelet edge extractor.

    Applies HL, LH, and HH Haar filters as a depthwise grouped convolution
    over all C input channels, then combines the three high-frequency
    sub-bands into a single edge-magnitude map of shape (B, C, H, W).

    Padding is chosen so spatial dimensions are preserved exactly.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.C = num_channels

        # --- Haar high-frequency filters (2x2) ---
        # HL: horizontal edges  (responds to vertical structure)
        HL = torch.tensor([[ 1.,  1.],
                            [-1., -1.]]) * 0.5
        # LH: vertical edges
        LH = torch.tensor([[ 1., -1.],
                            [ 1., -1.]]) * 0.5
        # HH: diagonal edges
        HH = torch.tensor([[ 1., -1.],
                            [-1.,  1.]]) * 0.5

        # Shape each filter to (1, 1, 2, 2), then tile over C channels
        # Final shape: (3*C, 1, 2, 2) for a grouped depthwise conv
        filters = torch.stack([HL, LH, HH], dim=0)          # (3, 2, 2)
        filters = filters.unsqueeze(1)                        # (3, 1, 2, 2)
        filters = filters.repeat(num_channels, 1, 1, 1)      # (3C, 1, 2, 2)

        # Register as a non-trainable buffer
        self.register_buffer('haar_filters', filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            edge_map: (B, C, H, W)  combined high-frequency magnitude
        """
        B, C, H, W = x.shape
        assert C == self.C, f"Expected {self.C} channels, got {C}"

        # Depthwise grouped conv: each channel filtered independently by 3 filters
        # groups = C ensures channel independence; we process all 3 sub-bands at once
        # by tiling filters over C (filter bank has shape (3C, 1, 2, 2))
        # We need to interleave channels so groups work correctly.
        # Strategy: repeat x to match the (3C) output channels of the filter bank.
        x_rep = x.repeat(1, 3, 1, 1)   # (B, 3C, H, W)

        # Pad to maintain H x W after 2x2 conv (pad right/bottom by 1)
        x_pad = F.pad(x_rep, (0, 1, 0, 1), mode='reflect')

        # Grouped depthwise conv: groups = 3C, each filter acts on one channel
        out = F.conv2d(x_pad, self.haar_filters, groups=3 * C)  # (B, 3C, H, W)

        # Split the three sub-bands and compute L2 magnitude
        HL_out, LH_out, HH_out = out.chunk(3, dim=1)             # each (B, C, H, W)
        edge_map = (HL_out.pow(2) + LH_out.pow(2) + HH_out.pow(2)).sqrt()

        return edge_map  # (B, C, H, W)


# ---------------------------------------------------------------------------
# 2. RepSR_Module  (Structural Re-parameterization Block)
# ---------------------------------------------------------------------------

class RepSR_Module(nn.Module):
    """
    Multi-branch conv block that collapses to a single 3x3 + dilated 3x3
    at inference time via reparameterize().

    Training branches:
        B1: 3x3 conv
        B2: 1x1 conv   (padded to match 3x3 output size)
        B3: identity   (only valid when in_channels == out_channels)
        B4: 3x3 dilated conv (dilation=2)  — kept separate at all times

    After reparameterize():
        forward = fused_3x3(x) + dilated_3x3(x)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self._reparameterized = False

        # --- Training branches ---
        self.conv3x3   = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv1x1   = nn.Conv2d(channels, channels, 1, bias=True)
        self.conv_dil  = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=True)

        # Identity branch: represented as a BN over the identity (common RepVGG trick)
        # We use a simple scale+bias parameter instead of BN for clarity & stability.
        self.id_scale  = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.id_bias   = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Placeholder for fused weight/bias (set after reparameterize())
        self.register_buffer('fused_weight', None)
        self.register_buffer('fused_bias',   None)

    # ------------------------------------------------------------------
    def _pad_1x1_to_3x3(self, w: torch.Tensor) -> torch.Tensor:
        """Zero-pad a (C_out, C_in, 1, 1) kernel to (C_out, C_in, 3, 3)."""
        return F.pad(w, [1, 1, 1, 1])

    def _identity_to_3x3(self) -> tuple:
        """Convert the identity branch to an equivalent 3x3 weight+bias."""
        C = self.channels
        # Identity kernel: 1 at center of each output-channel's own input-channel plane
        w_id = torch.zeros(C, C, 3, 3, device=self.id_scale.device,
                           dtype=self.id_scale.dtype)
        for i in range(C):
            w_id[i, i, 1, 1] = 1.0
        # Scale each output channel's kernel and bias
        w_id = w_id * self.id_scale.view(C, 1, 1, 1)
        b_id = self.id_bias.view(C)
        return w_id, b_id

    # ------------------------------------------------------------------
    def reparameterize(self):
        """
        Fuse B1 (3x3), B2 (1x1->3x3), and B3 (identity->3x3) into a single
        3x3 convolution.  B4 (dilated) is kept as-is.

        After this call, forward() uses only fused_3x3 + dilated_3x3.
        """
        if self._reparameterized:
            return  # Already done

        # Collect weights and biases
        w3, b3 = self.conv3x3.weight, self.conv3x3.bias
        w1, b1 = self._pad_1x1_to_3x3(self.conv1x1.weight), self.conv1x1.bias
        wi, bi = self._identity_to_3x3()

        # Fuse by summation (all branches share the same linear form)
        fused_w = w3 + w1 + wi
        fused_b = b3 + b1 + bi

        # Store as buffers for inference
        self.fused_weight = fused_w.detach()
        self.fused_bias   = fused_b.detach()

        # Remove training-only parameters to save memory
        del self.conv3x3, self.conv1x1, self.id_scale, self.id_bias

        self._reparameterized = True

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._reparameterized:
            # Fast inference path: fused 3x3 + dilated 3x3
            out  = F.conv2d(x, self.fused_weight, self.fused_bias, padding=1)
            out += self.conv_dil(x)
            return out
        else:
            # Training path: sum all four branches
            out  = self.conv3x3(x)
            out += F.pad(self.conv1x1(x), [0, 0, 0, 0])  # same shape, no extra pad needed
            out += x * self.id_scale + self.id_bias       # identity branch
            out += self.conv_dil(x)                       # dilated branch
            return out


# ---------------------------------------------------------------------------
# 3. WRR_Block  (Wavelet-Routed Reparameterization Block)
# ---------------------------------------------------------------------------

class WRR_Block(nn.Module):
    """
    Core block of WRRNet.

    Flow:
        x  ->  HaarDWT  ->  1x1 conv  ->  soft mask M  (Gumbel or threshold)
        x * M      ->  RepSR_Module  (heavy edge branch)
        x * (1-M)  ->  1x1 conv     (light background branch)
        output = edge_out + bg_out + x   (residual)
    """

    def __init__(self, channels: int, tau: float = 1.0):
        super().__init__()
        self.channels = channels
        self.tau = tau  # Gumbel temperature (anneal during training if desired)

        self.haar   = HaarDWTEdgePrior(channels)
        # Collapses C-channel edge map to a single spatial confidence score
        self.mask_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )

        self.edge_branch = RepSR_Module(channels)
        self.bg_branch   = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    # ------------------------------------------------------------------
    def _gumbel_binarize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Soft binarization via Gumbel-Softmax during training.
        Returns values in [0,1] that approximate {0,1}.
        """
        # Two-class Gumbel: class 1 = "edge", class 0 = "background"
        # logits shape: (B, 1, H, W)
        logits_2 = torch.cat([logits, -logits], dim=1)  # (B, 2, H, W)
        soft = F.gumbel_softmax(logits_2, tau=self.tau, hard=False, dim=1)
        return soft[:, 1:2, :, :]  # "edge" class probability (B,1,H,W)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step A: Wavelet edge prior
        edge_features = self.haar(x)          # (B, C, H, W)

        # Step B: Predict spatial routing mask
        raw_mask = self.mask_conv(edge_features)   # (B, 1, H, W)

        if self.training:
            M = self._gumbel_binarize(raw_mask)    # soft {0,1} mask
        else:
            M = (torch.sigmoid(raw_mask) > 0.5).float()  # hard binary mask

        # Step C: Dynamic routing
        edge_out = self.edge_branch(x * M)         # heavy path
        bg_out   = self.bg_branch(x * (1.0 - M))  # light path

        # Step D: Fuse + residual skip
        return edge_out + bg_out + x

    def reparameterize(self):
        """Delegate to inner RepSR module."""
        self.edge_branch.reparameterize()


# ---------------------------------------------------------------------------
# 4. WRRNet  (Full macro network)
# ---------------------------------------------------------------------------

class WRRNet(nn.Module):
    """
    Wavelet-Routed Reparameterization Network for image super-resolution.

    Args:
        scale       : upscaling factor (2, 3, or 4)
        num_channels: feature channels  (default 64)
        num_blocks  : number of WRR_Blocks in the deep stack (default 8)
        in_channels : input image channels (default 3 for RGB)
    """

    def __init__(self,
                 scale: int = 4,
                 num_channels: int = 64,
                 num_blocks: int = 8,
                 in_channels: int = 3):
        super().__init__()
        self.scale = scale

        # --- Shallow feature extraction ---
        self.shallow = nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=True)

        # --- Deep feature extraction: stack of WRR_Blocks ---
        self.blocks = nn.Sequential(
            *[WRR_Block(num_channels) for _ in range(num_blocks)]
        )

        # --- Upsampling head: expand -> pixel shuffle ---
        self.upsample_conv = nn.Conv2d(num_channels,
                                       (scale ** 2) * in_channels,
                                       kernel_size=3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Initialize weights
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  LR input image, values in [0, 1]
        Returns:
            sr: (B, 3, s*H, s*W)  super-resolved output
        """
        # Global skip: bilinearly upsample input to HR resolution
        skip = F.interpolate(x, scale_factor=self.scale,
                             mode='bilinear', align_corners=False)

        # Feature pipeline
        feat = self.shallow(x)          # (B, C, H, W)
        feat = self.blocks(feat)        # (B, C, H, W)
        feat = self.upsample_conv(feat) # (B, s^2*3, H, W)
        feat = self.pixel_shuffle(feat) # (B, 3, sH, sW)

        # Add global skip
        return feat + skip

    # ------------------------------------------------------------------
    def reparameterize(self):
        """
        Call once after training to collapse all RepSR multi-branch weights.
        After this, inference uses a single fused 3x3 + dilated 3x3 per block.
        """
        for block in self.blocks:
            block.reparameterize()
        print("[WRRNet] All RepSR_Module weights have been fused. "
              "Model is now in fast inference mode.")


# ---------------------------------------------------------------------------
# 5.  Training utilities
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    L1 pixel loss + L1 FFT frequency loss  (same as SSIU).
    lambda_fft=0.01 matches the paper's setting.
    """
    def __init__(self, lambda_fft: float = 0.01):
        super().__init__()
        self.lambda_fft = lambda_fft

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Pixel-domain L1
        l1 = F.l1_loss(pred, target)

        # Frequency-domain L1  (2D FFT magnitude)
        pred_fft   = torch.fft.rfft2(pred,   norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        lf = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

        return l1 + self.lambda_fft * lf


# ---------------------------------------------------------------------------
# 6.  Quick sanity check / parameter count
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    scale = 4
    model = WRRNet(scale=scale, num_channels=64, num_blocks=8)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"WRRNet  |  scale={scale}  |  params: {total_params/1e3:.1f} K")

    # Compare with SSIU target (~794 K for x4)
    print(f"SSIU baseline (x4): ~794 K params")

    # Dummy forward pass
    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        t0 = time.time()
        out = model(dummy)
        t1 = time.time()
    print(f"Output shape: {out.shape}  (expected [1, 3, {64*scale}, {64*scale}])")
    print(f"Forward pass time (CPU, single image): {(t1-t0)*1000:.1f} ms")

    # Test reparameterization
    print("\nRunning reparameterize()...")
    model.reparameterize()
    with torch.no_grad():
        t0 = time.time()
        out2 = model(dummy)
        t1 = time.time()
    print(f"Forward pass after reparameterize(): {(t1-t0)*1000:.1f} ms")

    # Verify numerical equivalence (should be near zero)
    # NOTE: reparameterized model forward will differ slightly due to
    # Gumbel in training mode, so we compare in eval mode
    diff = (out - out2).abs().max().item()
    print(f"Max output difference (train vs fused): {diff:.6f}  "
          f"({'OK' if diff < 1e-4 else 'CHECK'})")

    # Loss test
    criterion = CombinedLoss(lambda_fft=0.01)
    target = torch.randn_like(out2)
    loss = criterion(out2, target)
    print(f"\nCombined loss (sanity): {loss.item():.4f}")
