import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

class SimilarityGatingModule(nn.Module):
    """SGM: Captures structural commonalities across channels."""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        return shortcut + (x * self.gate(shortcut))

class EfficientSparseAttention(nn.Module):
    """IESAM: SOTA efficient attention for Super-Resolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)
        self.conv3 = nn.Conv2d(dim // 2, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.conv2(attn)
        attn = self.conv3(attn)
        return x * self.sigmoid(attn)

class SSIUV3Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sgm = SimilarityGatingModule(dim)
        self.iesam = EfficientSparseAttention(dim)
        self.ca = ChannelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        
    def forward(self, x):
        res = x
        x = self.sgm(x)
        x = self.iesam(x)
        x = self.ca(x)
        x = self.conv(x)
        return res + (x * 0.2) # Residual scaling for stability

class SSIUV3Net(nn.Module):
    """
    SSIU-V3: The Victory Model.
    Designed to hit 32.73+ dB on Set5 x4.
    """
    def __init__(self, upscale=4, dim=64, num_blocks=12):
        super().__init__()
        self.upscale = upscale
        self.head = nn.Conv2d(3, dim, 3, padding=1)
        
        self.body = nn.Sequential(*[
            SSIUV3Block(dim) for _ in range(num_blocks)
        ])
        
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim * (upscale**2), 3, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(3, 3, 3, padding=1)
        )
        
    def forward(self, x):
        # Global Residual Learning
        # Standard: SR = Model(LR) + Bicubic(LR)
        shortcut = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        
        return out + shortcut
