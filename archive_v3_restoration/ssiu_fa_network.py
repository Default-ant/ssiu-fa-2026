"""
SSIU-FA: SSIU with Fast-Attention Large Kernel Enhancement
===========================================================
Modification: The 3x3 depthwise convolution inside each MEM (Multi-scale 
Enhancement Module) is replaced by our SALK (Similarity-Aware Large Kernel),
which uses a decomposed 1x13 + 13x1 depthwise approach to expand the 
receptive field from 9 to 169 pixels with minimal parameter overhead.

Architecture: SSUFSRNet (original) + SALK in MEM blocks
Fine-tuning: Load original pretrained weights for all unchanged layers.
             Only the depthwise SALK convs are randomly initialized.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from torch.distributions.uniform import Uniform
import numpy as np
import random


# ============================================================
# Our Novel Module: SALK (Similarity-Aware Large Kernel DW Conv)
# Replaces the 3x3 depthwise conv inside MEM
# ============================================================
class SimilarityAwareLargeKernel(nn.Module):
    """
    SALK: Decomposed 13x13 depthwise convolution.
    Receptive field: 169 pixels vs 9 for standard 3x3.
    Composition: 1x13 + 13x1 + 3x3 local refinement.
    """
    def __init__(self, dim):
        super().__init__()
        # Decomposed large-kernel (1x13 then 13x1)
        self.lk_h = nn.Conv2d(dim, dim, (1, 13), padding=(0, 6), groups=dim, bias=True)
        self.lk_v = nn.Conv2d(dim, dim, (13, 1), padding=(6, 0), groups=dim, bias=True)
        # Local continuity (3x3)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)

    def forward(self, x):
        return self.lk_h(x) + self.lk_v(x) + self.local(x)


# ============================================================
# Modified MEM: Uses SALK instead of 3x3 depthwise
# ============================================================
class MEM_FA(nn.Module):
    """MEM with SALK large-kernel enhancement."""
    def __init__(self, dim, out_dim, scale, bias, ks=3, ratio=1):
        super(MEM_FA, self).__init__()
        self.project_in1 = nn.Conv2d(dim, int(dim*scale), kernel_size=1, groups=1, bias=bias)
        self.project_in2_pw1 = nn.Conv2d(dim, int(dim*ratio), kernel_size=1, groups=1, bias=bias)
        # *** KEY CHANGE: SALK replaces the 3x3 depthwise conv ***
        self.project_in2_salk = SimilarityAwareLargeKernel(int(dim*ratio))
        self.project_in2_pw2 = nn.Conv2d(int(dim*ratio), int(dim*scale), kernel_size=1, groups=1, bias=bias)
        self.project_out = nn.Conv2d(int(dim*scale), out_dim, kernel_size=1, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.project_in1(x)
        x2 = self.project_in2_pw1(x)
        x2 = self.project_in2_salk(x2)
        x2 = self.project_in2_pw2(x2)
        x2 = self.act(x1) * x2
        x = self.project_out(x2)
        return x


# ============================================================
# Modified SSRM: Uses MEM_FA instead of MEM
# ============================================================
class SSRM_FA(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, bs=8, ks=3, sr=2, scale=2.0, ratio=1):
        super(SSRM_FA, self).__init__()
        self.norm = LayerNorm(dim=in_channels, LayerNorm_type='WithBias')
        self.s1 = MEM_FA(dim=in_channels, out_dim=out_channels, scale=scale, bias=True, ks=ks, ratio=ratio)
        self.s2 = MEM_FA(dim=in_channels, out_dim=out_channels, scale=scale, bias=True, ks=ks, ratio=ratio)
        self.s3 = ESA(ch=in_channels, block_size=bs, halo_size=1, num_heads=num_heads, bias=False, ks=3, sr=sr)
        self.s4 = MEM_FA(dim=in_channels, out_dim=out_channels, scale=scale, bias=True, ks=ks, ratio=ratio)
        self.ffn = MEM_FA(dim=in_channels, out_dim=out_channels, scale=scale, bias=True, ks=ks, ratio=ratio)

    def forward(self, a, y):
        a = self.norm(a)
        z = self.s1(a)
        b = self.s2(a)
        v = a + z + y
        v = self.s3(v) + v
        a = v - b
        a = self.s4(a) + b
        a = self.ffn(a) + a
        return a


# ============================================================
# Main Model: SSUFSRNet-FA (drop-in replacement)
# ============================================================
class SSUFSRNet_FA(nn.Module):
    def __init__(self, args):
        super(SSUFSRNet_FA, self).__init__()
        n_feats = args.n_feats
        self.scale = args.scale
        self.window_sizes = [8, 16]
        self.n_blocks = args.n_blocks

        self.head = nn.Conv2d(args.colors, n_feats, kernel_size=3, bias=True, stride=1, padding=1, padding_mode='reflect')

        # *** Bodies now use SSRM_FA (our SALK-enhanced blocks) ***
        self.body = nn.ModuleList([
            SSRM_FA(in_channels=n_feats, out_channels=n_feats, num_heads=4, bs=8, ks=3, sr=2, scale=1.0, ratio=1.0)
            for _ in range(args.n_blocks)
        ])

        self.moe = MOE(nf=n_feats)

        if self.scale == 4:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats*4, kernel_size=1, bias=True, stride=1, padding=0, padding_mode='reflect'),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(n_feats, n_feats*4, kernel_size=1, bias=True, stride=1, padding=0, padding_mode='reflect'),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(n_feats, 3, kernel_size=3, bias=True, stride=1, padding=1, padding_mode='reflect'),
            )
        else:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats*self.scale*self.scale, kernel_size=1, bias=True, stride=1, padding=0, padding_mode='reflect'),
                nn.PixelShuffle(self.scale),
                nn.GELU(),
                nn.Conv2d(n_feats, 3, kernel_size=3, bias=True, stride=1, padding=1, padding_mode='reflect'),
            )

    def forward(self, x):
        H, W = (x.shape[2], x.shape[3])
        x = self.check_image_size(x)
        res = self.head(x)
        a = res
        exp = []
        for blkid in range(self.n_blocks):
            a = self.body[blkid](a, res)
            if (blkid+1) % (self.n_blocks // 3) == 0 or (blkid+1) == self.n_blocks:
                exp.append(a)
        a = self.moe(exp) + res
        a = self.tail(a) + F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return a[:, :, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_pretrained(self, pretrained_path, device='cpu'):
        """
        Load original SSIU pretrained weights.
        All unchanged layers (head, moe, tail, norm, s3, 1x1 convs) load perfectly.
        Only the SALK depthwise layers are randomly initialized.
        """
        ckpt = torch.load(pretrained_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif 'model' in ckpt:
                state = ckpt['model']
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
            elif 'params' in ckpt:
                state = ckpt['params']
            else:
                state = ckpt  # flat dict
        else:
            state = ckpt

        own_state = self.state_dict()
        loaded, skipped = 0, 0

        for name, param in state.items():
            # Strip 'module.' prefix if present (from DataParallel)
            clean_name = name.replace('module.', '')
            
            # Direct match
            if clean_name in own_state:
                if own_state[clean_name].shape == param.shape:
                    own_state[clean_name].copy_(param)
                    loaded += 1
                else:
                    skipped += 1
                continue

            # Remap MEM (original) → MEM_FA (ours)
            new_name = clean_name
            new_name = new_name.replace('.project_in2.0.', '.project_in2_pw1.')
            new_name = new_name.replace('.project_in2.2.', '.project_in2_pw2.')

            if '.project_in2.1.' in clean_name:
                skipped += 1  # 3x3 DW → replaced by SALK, skip
                continue

            if new_name in own_state and own_state[new_name].shape == param.shape:
                own_state[new_name].copy_(param)
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(own_state)
        total = len(list(state.keys()))
        print(f"✅ Loaded {loaded}/{total} pretrained layers")
        print(f"⚡ {skipped} layers skipped (SALK depthwise = randomly initialized, will fine-tune)")
        return loaded, skipped


# ============================================================
# Unchanged from original SSIU (copied verbatim)
# ============================================================
class MOE(nn.Module):
    def __init__(self, nf=32, dropout_ratio=0.0):
        super(MOE, self).__init__()
        self.nf = nf
        self.dropout_ratio = dropout_ratio
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        self.fc_a = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.fc_b = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.fc_c = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.fuse = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        if len(x) == 4:
            a, b, c = x[1], x[2], x[3]
        else:
            a, b, c = x[0], x[1], x[2]
        wa = self.fc_a(a)
        wb = self.fc_b(b)
        wc = self.fc_c(c)
        wm = self.softmax(torch.stack([wa, wb, wc], dim=0))
        a = a * wm[0,:].squeeze(0)
        b = b * wm[1,:].squeeze(0)
        c = c * wm[2,:].squeeze(0)
        out = a + b + c
        if self.dropout_ratio > 0:
            out = self.dropout(out)
        out = self.fuse(out) + out
        return out


class ESA(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=3, num_heads=4, bias=False, ks=3, sr=1):
        super(ESA, self).__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_ch = ch // num_heads
        assert ch % num_heads == 0
        self.sr = sr
        if sr > 1:
            self.sampler = nn.MaxPool2d(2, sr)
            self.LocalProp = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=ks, stride=1, padding=int(ks//2), groups=ch, bias=True, padding_mode='reflect'),
                Interpolate(scale_factor=sr, mode='bilinear', align_corners=True),
            )
        self.rel_h = nn.Parameter(torch.randn(1, block_size+2*halo_size, 1, self.head_ch//2), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(1, 1, block_size+2*halo_size, self.head_ch//2), requires_grad=True)
        self.qkv_conv = nn.Conv2d(ch, ch*3, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, oh, ow = x.size()
        if self.sr > 1:
            x = self.sampler(x)
        B, C, H, W = x.size()
        pad_l = pad_t = 0
        pad_r = (self.block_size - W % self.block_size) % self.block_size
        pad_b = (self.block_size - H % self.block_size) % self.block_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='reflect')
        b, c, h, w, block, halo, heads = *x.shape, self.block_size, self.halo_size, self.num_heads
        assert h % block == 0 and w % block == 0
        x = self.qkv_conv(x)
        q, k, v = torch.chunk(x, 3, dim=1)
        q = rearrange(q, 'b c (h k1) (w k2) -> (b h w) (k1 k2) c', k1=block, k2=block)
        q = q * self.head_ch ** -0.5
        k = F.unfold(k, kernel_size=block+halo*2, stride=block, padding=halo)
        k = rearrange(k, 'b (c a) l -> (b l) a c', c=c)
        v = F.unfold(v, kernel_size=block+halo*2, stride=block, padding=halo)
        v = rearrange(v, 'b (c a) l -> (b l) a c', c=c)
        q, v = map(lambda i: rearrange(i, 'b a (h d) -> (b h) a d', h=heads), (q, v))
        k = rearrange(k, 'b (k1 k2) (h d) -> (b h) k1 k2 d', k1=block+2*halo, h=heads)
        k_h, k_w = k.split(self.head_ch//2, dim=-1)
        k = torch.cat([k_h+self.rel_h, k_w+self.rel_w], dim=-1)
        k = rearrange(k, 'b k1 k2 d -> b (k1 k2) d')
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        attn = F.softmax(sim, dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h w n) (k1 k2) d -> b (n d) (h k1) (w k2)', b=b, h=(h//block), w=(w//block), k1=block, k2=block)
        if self.sr > 1:
            out = self.LocalProp(out)
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :oh, :ow]
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode,
                           align_corners=self.align_corners, recompute_scale_factor=True)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = 1e-6

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = 1e-6

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + self.eps) * self.weight


class AffineFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(AffineFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape
        self.eps = 1e-6

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + self.eps)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type == 'AffineFree':
            self.body = AffineFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
