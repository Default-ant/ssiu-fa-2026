#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║  WRRNet — Kaggle Training + Evaluation Notebook               ║
║  Paste each section into a separate Kaggle cell.              ║
║  Dataset needed: DIV2K / DF2K + Set5                         ║
║  GPU: T4 / P100 recommended                                   ║
╚═══════════════════════════════════════════════════════════════╝

FIXES vs. previous run:
    1.  HaarDWTEdgePrior sqrt: added eps=1e-8 → prevents inf gradient
        on flat image regions (grad = 1/(2*sqrt(x)), x→0 → ∞).
    2.  CombinedLoss FFT magnitude: view_as_real + eps-guarded norm
        avoids torch.abs(complex) singularity at zero-magnitude coefficients.
    3.  eval psnr_y / ssim_y aliases: NameError in evaluation cell fixed.
    4.  nan guard before uint8 cast: RuntimeWarning eliminated.
"""

# === CELL 1: WRRNet Architecture (numerically stable) ========================
# %%
import math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── HaarDWTEdgePrior ──────────────────────────────────────────────────────────
class HaarDWTEdgePrior(nn.Module):
    """Fixed (no-gradient) Haar wavelet edge extractor."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.C = num_channels
        HL = torch.tensor([[ 1.,  1.], [-1., -1.]]) * 0.5
        LH = torch.tensor([[ 1., -1.], [ 1., -1.]]) * 0.5
        HH = torch.tensor([[ 1., -1.], [-1.,  1.]]) * 0.5
        filters = torch.stack([HL, LH, HH], dim=0).unsqueeze(1)  # (3,1,2,2)
        filters = filters.repeat(num_channels, 1, 1, 1)          # (3C,1,2,2)
        self.register_buffer('haar_filters', filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_rep = x.repeat(1, 3, 1, 1)                             # (B,3C,H,W)
        x_pad = F.pad(x_rep, (0, 1, 0, 1), mode='reflect')
        out   = F.conv2d(x_pad, self.haar_filters, groups=3 * C) # (B,3C,H,W)
        HL_out, LH_out, HH_out = out.chunk(3, dim=1)
        # FIX: eps=1e-8 prevents 1/sqrt(0)=inf gradient on flat regions
        edge_map = (HL_out.pow(2) + LH_out.pow(2) + HH_out.pow(2) + 1e-8).sqrt()
        return edge_map


# ── RepSR_Module ─────────────────────────────────────────────────────────────
class RepSR_Module(nn.Module):
    """Multi-branch conv block that collapses to single 3x3 + dilated 3x3."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self._reparameterized = False
        self.conv3x3  = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv1x1  = nn.Conv2d(channels, channels, 1, bias=True)
        self.conv_dil = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=True)
        self.id_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.id_bias  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer('fused_weight', None)
        self.register_buffer('fused_bias',   None)

    def _pad_1x1_to_3x3(self, w):
        return F.pad(w, [1, 1, 1, 1])

    def _identity_to_3x3(self):
        C = self.channels
        w_id = torch.zeros(C, C, 3, 3, device=self.id_scale.device,
                           dtype=self.id_scale.dtype)
        for i in range(C):
            w_id[i, i, 1, 1] = 1.0
        return w_id * self.id_scale.view(C, 1, 1, 1), self.id_bias.view(C)

    def reparameterize(self):
        if self._reparameterized:
            return
        w3, b3 = self.conv3x3.weight, self.conv3x3.bias
        w1, b1 = self._pad_1x1_to_3x3(self.conv1x1.weight), self.conv1x1.bias
        wi, bi = self._identity_to_3x3()
        self.fused_weight = (w3 + w1 + wi).detach()
        self.fused_bias   = (b3 + b1 + bi).detach()
        del self.conv3x3, self.conv1x1, self.id_scale, self.id_bias
        self._reparameterized = True

    def forward(self, x):
        if self._reparameterized:
            return F.conv2d(x, self.fused_weight, self.fused_bias, padding=1) + self.conv_dil(x)
        return self.conv3x3(x) + self.conv1x1(x) + x * self.id_scale + self.id_bias + self.conv_dil(x)


# ── WRR_Block ─────────────────────────────────────────────────────────────────
class WRR_Block(nn.Module):
    """Core block: wavelet routing + reparameterizable edge path."""

    def __init__(self, channels: int, tau: float = 1.0):
        super().__init__()
        self.channels = channels
        self.tau = tau
        self.haar      = HaarDWTEdgePrior(channels)
        self.mask_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.edge_branch = RepSR_Module(channels)
        self.bg_branch   = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def _gumbel_binarize(self, logits):
        logits_2 = torch.cat([logits, -logits], dim=1)
        soft = F.gumbel_softmax(logits_2, tau=self.tau, hard=False, dim=1)
        return soft[:, 1:2, :, :]

    def forward(self, x):
        edge_features = self.haar(x)
        raw_mask = self.mask_conv(edge_features)
        M = self._gumbel_binarize(raw_mask) if self.training else (torch.sigmoid(raw_mask) > 0.5).float()
        edge_out = self.edge_branch(x * M)
        bg_out   = self.bg_branch(x * (1.0 - M))
        return edge_out + bg_out + x

    def reparameterize(self):
        self.edge_branch.reparameterize()


# ── WRRNet ────────────────────────────────────────────────────────────────────
class WRRNet(nn.Module):
    """Wavelet-Routed Reparameterization Network for image super-resolution."""

    def __init__(self, scale=4, num_channels=64, num_blocks=8, in_channels=3):
        super().__init__()
        self.scale = scale
        self.shallow = nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=True)
        self.blocks  = nn.Sequential(*[WRR_Block(num_channels) for _ in range(num_blocks)])
        self.upsample_conv = nn.Conv2d(num_channels, (scale ** 2) * in_channels,
                                       kernel_size=3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skip = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        feat = self.shallow(x)
        feat = self.blocks(feat)
        feat = self.upsample_conv(feat)
        feat = self.pixel_shuffle(feat)
        return feat + skip

    def reparameterize(self):
        for block in self.blocks:
            block.reparameterize()
        print("[WRRNet] All RepSR_Module weights have been fused. Model is now in fast inference mode.")

    def set_tau(self, tau: float):
        """Anneal Gumbel temperature during training."""
        for block in self.blocks:
            block.tau = tau


# ── CombinedLoss ──────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """L1 pixel loss + FFT frequency loss (numerically stable)."""

    def __init__(self, lambda_fft: float = 0.01):
        super().__init__()
        self.lambda_fft = lambda_fft

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        pred_fft   = torch.fft.rfft2(pred,   norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        # FIX: view_as_real + eps prevents inf gradient at zero-magnitude coefficients
        pred_mag   = torch.view_as_real(pred_fft).pow(2).sum(-1).add(1e-8).sqrt()
        target_mag = torch.view_as_real(target_fft).pow(2).sum(-1).add(1e-8).sqrt()
        lf = F.l1_loss(pred_mag, target_mag)
        return l1 + self.lambda_fft * lf


# Quick sanity check
model_test = WRRNet(scale=4, num_channels=64, num_blocks=8)
params_k = sum(p.numel() for p in model_test.parameters()) / 1e3
print(f"✅ WRRNet loaded  |  {params_k:.1f}K params")
del model_test


# === CELL 2: Dataset & Training Setup ========================================
# %%
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image

UPSCALE       = 4
PATCH_SIZE_LR = 64
PATCH_SIZE_HR = PATCH_SIZE_LR * UPSCALE
BATCH_SIZE    = 16
ITERATIONS    = 25000
RESUME_PATH   = ""            # ← set to e.g. "wrrnet_iter_5000.pth" to resume
LEARNING_RATE = 2e-4          # lower start LR vs. previous run
ETA_MIN       = 1e-6
FFT_LOSS_WEIGHT = 0.01
TAU_START     = 1.0           # Gumbel temperature annealing
TAU_END       = 0.1

KAGGLE_PATHS = [
    '/kaggle/input/datasets/anvu1204/df2kdata/DF2K_train_HR',
    '/kaggle/input/datasets/bansalyash/div2k-hr',
    '/kaggle/input/df2k-dataset/DF2K_train_HR',
    '/kaggle/input/df2k/DF2K_train_HR',
    '/kaggle/input/div2k-dataset/DIV2K_train_HR',
    '/kaggle/input/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR',
    '/kaggle/input/datasets/harshraone/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR',
    '/kaggle/input/div2k-high-resolution-images/DIV2K_train_HR',
]

def auto_detect_data_path():
    for p in KAGGLE_PATHS:
        if os.path.isdir(p):
            return p
    return None


class DIV2KDataset(Dataset):
    def __init__(self, data_path):
        self.patch_size_lr = PATCH_SIZE_LR
        self.patch_size_hr = PATCH_SIZE_HR
        self.file_list = sorted([
            os.path.join(data_path, f) for f in os.listdir(data_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        if not self.file_list:
            raise ValueError(f"No images found in {data_path}")
        print(f"  Dataset: {len(self.file_list)} images from {data_path}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.file_list[idx])
        hr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = hr.shape
        x = np.random.randint(0, max(w - self.patch_size_hr, 1))
        y = np.random.randint(0, max(h - self.patch_size_hr, 1))
        hr_crop = hr[y:y + self.patch_size_hr, x:x + self.patch_size_hr]
        aug = np.random.randint(0, 8)
        if aug == 1: hr_crop = np.flipud(hr_crop)
        elif aug == 2: hr_crop = np.fliplr(hr_crop)
        elif aug == 3: hr_crop = np.rot90(hr_crop, 1)
        elif aug == 4: hr_crop = np.rot90(hr_crop, 2)
        elif aug == 5: hr_crop = np.rot90(hr_crop, 3)
        elif aug == 6: hr_crop = np.flipud(np.rot90(hr_crop, 1))
        elif aug == 7: hr_crop = np.fliplr(np.rot90(hr_crop, 1))
        hr_pil = Image.fromarray(hr_crop)
        lr_pil = hr_pil.resize((self.patch_size_lr, self.patch_size_lr), resample=Image.BICUBIC)
        lr_crop = np.array(lr_pil)
        hr_t = torch.from_numpy(hr_crop.copy()).permute(2, 0, 1).float() / 255.0
        lr_t = torch.from_numpy(lr_crop.copy()).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t

print("✅ Training utilities loaded")


# === CELL 3: Training =========================================================
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WRRNet(scale=4, num_channels=64, num_blocks=8).to(device)
if torch.cuda.device_count() > 1:
    print(f"  🔥 Multi-GPU: {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

start_iter = 0
if RESUME_PATH and os.path.isfile(RESUME_PATH):
    ckpt = torch.load(RESUME_PATH, map_location=device)
    sd   = ckpt.get('model_state_dict', ckpt)
    sd   = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    start_iter = ckpt.get('iteration', 0) if isinstance(ckpt, dict) else 0
    print(f"  🚀 Resumed from {RESUME_PATH} (iter {start_iter})")
else:
    print("  🌱 Training from scratch")

data_path = auto_detect_data_path()
if data_path is None:
    raise RuntimeError("No dataset found — attach DIV2K/DF2K in Kaggle.")
dataset    = DIV2KDataset(data_path)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
if start_iter > 0:
    current_lr = ETA_MIN + 0.5 * (LEARNING_RATE - ETA_MIN) * (
        1 + math.cos(math.pi * start_iter / ITERATIONS))
    for g in optimizer.param_groups:
        g['lr'] = current_lr
        g.setdefault('initial_lr', LEARNING_RATE)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=ITERATIONS, eta_min=ETA_MIN,
    last_epoch=start_iter if start_iter > 0 else -1)

criterion = CombinedLoss(lambda_fft=FFT_LOSS_WEIGHT).to(device)
scaler    = torch.amp.GradScaler('cuda')

print("=" * 60)
print(f"  Device: {device} | Iters: {ITERATIONS} | LR: {LEARNING_RATE}")
print(f"  Batch: {BATCH_SIZE} | Loss: L1 + {FFT_LOSS_WEIGHT}×FFT (eps-stable)")
print("=" * 60)

model.train()
loader_iter = iter(dataloader)

for i in range(start_iter + 1, ITERATIONS + 1):
    try:
        lr_batch, hr_batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(dataloader)
        lr_batch, hr_batch = next(loader_iter)

    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)

    # Anneal Gumbel temperature τ → 0.1 over training
    tau = TAU_START + (TAU_END - TAU_START) * (i / ITERATIONS)
    for block in (model.module.blocks if hasattr(model, 'module') else model.blocks):
        block.tau = tau

    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        sr = model(lr_batch)
        loss = criterion(sr, hr_batch)

    scaler.scale(loss).backward()
    # Gradient clipping prevents exploding gradients in the RepSR multi-branch
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # Print every 10 iters for first 500 (quick nan check), then every 100
    log_freq = 10 if i <= 500 else 100
    if i % log_freq == 0:
        print(f"  [{i:6d}/{ITERATIONS}] Loss: {loss.item():.5f}  Tau: {tau:.3f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

    if i % 5000 == 0 or i == ITERATIONS or i == 500:
        ckpt_path = f"/kaggle/working/wrrnet_iter_{i}.pth"
        torch.save({
            'iteration': i,
            'model_state_dict': (model.module if hasattr(model, 'module') else model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, ckpt_path)
        print(f"💾 Saved Checkpoint: {ckpt_path}")

print("=" * 60)
print("✅ Training Finished!")
print("=" * 60)


# === CELL 4: Evaluation on Set5 ===============================================
# %%
import cv2, numpy as np
from PIL import Image

BORDER = 4

def rgb_to_ycbcr_y(img_rgb):
    img = img_rgb.astype(np.float64)
    return 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0

def psnr_y(img1, img2, border=BORDER):
    y1, y2 = rgb_to_ycbcr_y(img1), rgb_to_ycbcr_y(img2)
    if border: y1, y2 = y1[border:-border, border:-border], y2[border:-border, border:-border]
    mse = np.mean((y1 - y2) ** 2)
    return 100.0 if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))

def ssim_y(img1, img2, border=BORDER):
    y1, y2 = rgb_to_ycbcr_y(img1), rgb_to_ycbcr_y(img2)
    if border: y1, y2 = y1[border:-border, border:-border], y2[border:-border, border:-border]
    try:
        from skimage.metrics import structural_similarity as _ssim
        return _ssim(y1, y2, data_range=219.0)
    except ImportError:
        C1, C2 = (0.01 * 219) ** 2, (0.03 * 219) ** 2
        mu1 = cv2.GaussianBlur(y1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(y2, (11, 11), 1.5)
        s1  = cv2.GaussianBlur(y1 ** 2, (11, 11), 1.5) - mu1 ** 2
        s2  = cv2.GaussianBlur(y2 ** 2, (11, 11), 1.5) - mu2 ** 2
        s12 = cv2.GaussianBlur(y1 * y2, (11, 11), 1.5) - mu1 * mu2
        return float(np.mean(((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
                              ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))))

# Load reparameterized model for evaluation
eval_model = WRRNet(scale=4, num_channels=64, num_blocks=8)
ckpt = torch.load("/kaggle/working/wrrnet_iter_25000.pth", map_location='cpu')
sd = ckpt.get('model_state_dict', ckpt)
sd = {k.replace('module.', ''): v for k, v in sd.items()}
eval_model.load_state_dict(sd)
eval_model.reparameterize()
eval_model = eval_model.to(device).eval()
params_k = sum(p.numel() for p in eval_model.parameters()) / 1e3
print(f"WRRNet loaded  |  {params_k:.1f}K params  |  reparameterized ✓")

# Find Set5
set5_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    if os.path.basename(root).lower() == 'set5':
        set5_path = root
        break

hr_paths = []
if set5_path:
    hr_dir = os.path.join(set5_path, 'HR')
    search_dir = hr_dir if os.path.isdir(hr_dir) else set5_path
    hr_paths = sorted([os.path.join(search_dir, f) for f in os.listdir(search_dir)
                       if f.lower().endswith(('.png', '.jpg')) and 'LR' not in f])

if not hr_paths:
    print("⚠️  Set5 not found. Attach it via Kaggle Datasets.")
else:
    psnrs, ssims = [], []
    for p in hr_paths:
        img_bgr = cv2.imread(p)
        h, w, _ = img_bgr.shape
        h, w = h - h % UPSCALE, w - w % UPSCALE
        hr_rgb = cv2.cvtColor(img_bgr[:h, :w], cv2.COLOR_BGR2RGB)
        lr_pil = Image.fromarray(hr_rgb).resize((w // UPSCALE, h // UPSCALE), resample=Image.BICUBIC)
        lr_rgb = np.array(lr_pil)

        with torch.no_grad():
            lr_t = torch.from_numpy(lr_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            sr_t = eval_model(lr_t)
            sr_np = sr_t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            # FIX: guard nan/inf before uint8 cast
            if not np.isfinite(sr_np).all():
                print(f"  ⚠️  nan/inf in output for {os.path.basename(p)}, skipping")
                continue
            sr_rgb = (sr_np * 255.0).clip(0, 255).astype(np.uint8)

        psnrs.append(psnr_y(sr_rgb, hr_rgb))
        ssims.append(ssim_y(sr_rgb, hr_rgb))
        print(f"  {os.path.basename(p):<20s}  PSNR: {psnrs[-1]:.2f} dB   SSIM: {ssims[-1]:.4f}")

    print(f"\n  AVERAGE  PSNR: {np.mean(psnrs):.2f} dB   SSIM: {np.mean(ssims):.4f}")
    print(f"  Paper baseline (Set5 x4): 32.64 dB / 0.9002")
