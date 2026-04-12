#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  SSIU-FA 24-Block Training + Evaluation — Kaggle Notebook       ║
║  Copy each section into a separate Kaggle cell.                 ║
║  Datasets needed: DIV2K, Set5 (HR/LR)                          ║
║  GPU: T4/P100 recommended                                      ║
╚══════════════════════════════════════════════════════════════════╝

CELL MARKERS: Search for "# === CELL" to find cell boundaries.
"""

# === CELL 1: Model Architecture (ssiu_improved.py) ===========================
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Fixed Configuration ───────────────────────────────────────────────────────
SCALE = 4
EMBED_DIM = 64
NUM_BLOCKS = 48
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
        self.lk1 = nn.Conv2d(dim, dim, (1, 13), padding=(0, 6), groups=dim)
        self.lk2 = nn.Conv2d(dim, dim, (13, 1), padding=(6, 0), groups=dim)
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
    Architecture: 24 Blocks, 64 channels, x4 upscale.
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
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear',
                             align_corners=False)
        f_in = self.conv_in(x)
        feat = f_in
        for layer in self.layers:
            feat = layer(feat)
        out = self.upsampler(self.conv_after(feat) + f_in)
        return out + base


# Verify architecture
model_test = ImprovedSSIUNet()
params_k = sum(p.numel() for p in model_test.parameters()) / 1e3
print(f"✅ SSIU-FA Architecture Loaded")
print(f"   Blocks: {NUM_BLOCKS} | Channels: {EMBED_DIM} | Params: {params_k:.1f}K")
del model_test


# === CELL 2: Dataset & Training Setup ========================================
# %%
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import math
from PIL import Image

# ─── Training Configuration ───────────────────────────────────────────────────
UPSCALE = 4
PATCH_SIZE_LR = 64
PATCH_SIZE_HR = PATCH_SIZE_LR * UPSCALE   # 256
BATCH_SIZE = 32
ITERATIONS = 25000
RESUME_PATH = "ssiu_fa_48b_start.pth"  # This will be generated dynamically on Kaggle
LEARNING_RATE = 1e-3
ETA_MIN = 1e-6
FFT_LOSS_WEIGHT = 0.05
# ────────────────────────────────────────────────────────────────────────────────

# Auto-detect training data on Kaggle (DF2K preferred over DIV2K)
KAGGLE_PATHS = [
    # User-provided exact paths
    '/kaggle/input/datasets/anvu1204/df2kdata/DF2K_train_HR',
    '/kaggle/input/datasets/bansalyash/div2k-hr',
    # DF2K (3450 images — preferred for best results)
    '/kaggle/input/df2k-dataset/DF2K_train_HR',
    '/kaggle/input/df2k/DF2K_train_HR',
    # DIV2K fallback (800 images)
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
    """DIV2K training dataset: HR → PIL bicubic LR, RGB, [0,1]."""
    def __init__(self, data_path):
        super().__init__()
        self.patch_size_lr = PATCH_SIZE_LR
        self.patch_size_hr = PATCH_SIZE_HR

        self.file_list = sorted([
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        if len(self.file_list) == 0:
            raise ValueError(f"No images found in {data_path}.")
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
        lr_pil = hr_pil.resize(
            (self.patch_size_lr, self.patch_size_lr),
            resample=Image.BICUBIC
        )
        lr_crop = np.array(lr_pil)

        hr_t = torch.from_numpy(hr_crop.copy()).permute(2, 0, 1).float() / 255.0
        lr_t = torch.from_numpy(lr_crop.copy()).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def frequency_loss(sr, hr):
    sr_fft = torch.fft.rfft2(sr, norm='ortho')
    hr_fft = torch.fft.rfft2(hr, norm='ortho')
    return torch.mean(torch.abs(sr_fft - hr_fft))


print("✅ Training utilities loaded")
print(f"   Iterations: {ITERATIONS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")


# === CELL 3: Training ========================================================
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("  SSIU-FA 24-BLOCK TRAINING — x4 Super-Resolution")
print("=" * 60)
print(f"  Device      : {device}")
print(f"  Scale       : x{UPSCALE}")
print(f"  Channels    : {EMBED_DIM}")
print(f"  Blocks      : {NUM_BLOCKS}")
print(f"  Patch (LR)  : {PATCH_SIZE_LR}×{PATCH_SIZE_LR}")
print(f"  Patch (HR)  : {PATCH_SIZE_HR}×{PATCH_SIZE_HR}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Iterations  : {ITERATIONS}")
print(f"  LR          : {LEARNING_RATE} → {ETA_MIN} (cosine)")
print(f"  Loss        : Charbonnier + {FFT_LOSS_WEIGHT}×FFT")
print(f"  AMP         : Enabled")

# Model
model = ImprovedSSIUNet(upscale=UPSCALE, embed_dim=EMBED_DIM,
                        num_blocks=NUM_BLOCKS).to(device)
# Resume weights if provided (Now natively bundled in Git)
start_iter = 0
if RESUME_PATH and os.path.isfile(RESUME_PATH):
    print(f"  🚀 Resuming from: {RESUME_PATH}")
    checkpoint = torch.load(RESUME_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        # If it's the warm start, we treat it as iter 0 but with better knowledge
        if "start" in RESUME_PATH:
            start_iter = 0
            print("  🔥 Warm Start detected: Bootstrapping 28-block model with 24-block knowledge.")
        else:
            start_iter = checkpoint.get('iteration', 0)
    else:
        model.load_state_dict(checkpoint)
        if "start" in RESUME_PATH:
            start_iter = 0
            print("  🔥 Warm Start detected: Bootstrapping 28-block model with 24-block knowledge.")
        elif '20000' in RESUME_PATH: start_iter = 20000
    print(f"  Continuing from iteration: {start_iter}")
else:
    print("  🌱 Starting from scratch")

# Dataset
data_path = auto_detect_data_path()
if data_path is None:
    raise RuntimeError(
        "ERROR: DIV2K not found! Please attach the DIV2K dataset to this notebook.\n"
        "Go to: Add Data → Search 'div2k' → Add 'DIV2K Dataset'"
    )
dataset = DIV2KDataset(data_path)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True
)

# Optimizer & Scheduler
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

# Mandatory: Force initial LR calculation if resuming to prevent "LR jump"
current_lr = LEARNING_RATE
if start_iter > 0:
    current_lr = ETA_MIN + 0.5 * (LEARNING_RATE - ETA_MIN) * (
        1 + math.cos(math.pi * start_iter / ITERATIONS)
    )

for group in optimizer.param_groups:
    group['lr'] = current_lr
    group.setdefault('initial_lr', LEARNING_RATE)

# Initialize scheduler with last_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=ITERATIONS, eta_min=ETA_MIN, 
    last_epoch=start_iter if start_iter > 0 else -1
)

criterion = CharbonnierLoss()
scaler = torch.amp.GradScaler('cuda')

print("=" * 60)
print(f"  Target Iter : {ITERATIONS}")
print(f"  Starting LR : {optimizer.param_groups[0]['lr']:.6f}")
print("  🚀 Training started...")
print("=" * 60)

# Training Loop
model.train()
loader_iter = iter(dataloader)

for i in range(start_iter + 1, ITERATIONS + 1):
    try:
        lr, hr = next(loader_iter)
    except StopIteration:
        loader_iter = iter(dataloader)
        lr, hr = next(loader_iter)

    lr, hr = lr.to(device), hr.to(device)

    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        sr = model(lr)
        loss_char = criterion(sr, hr)
        loss_fft = frequency_loss(sr, hr)
        loss = loss_char + FFT_LOSS_WEIGHT * loss_fft

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if i % 100 == 0:
        print(f"  [{i:6d}/{ITERATIONS}] "
              f"Loss: {loss.item():.5f} "
              f"(Char: {loss_char.item():.5f} | FFT: {loss_fft.item():.5f}) "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    if i % 5000 == 0 or i == ITERATIONS:
        ckpt_path = f"ssiu_fa_24b_iter_{i}.pth"
        save_data = {
            'iteration': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(save_data, ckpt_path)
        print(f"  >>> Checkpoint saved: {ckpt_path}")

# Final save (clean weights only)
final_path = "ssiu_fa_24b_final.pth"
torch.save(model.state_dict(), final_path)
print(f"\n  ✅ TRAINING COMPLETE. Final weights: {final_path}")
print("=" * 60)


# === CELL 4: Evaluation on Set5 ==============================================
# %%
import cv2
import numpy as np
from PIL import Image

BORDER = 4

# Auto-detect Set5
SET5_PATHS = [
    '/kaggle/input/set5-hr-lr/Set5',
    '/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5',
    '/kaggle/input/set5/Set5/HR',
    '/kaggle/input/set5/Set5',
]

def find_set5():
    for p in SET5_PATHS:
        if os.path.isdir(p):
            return p
    return None

def rgb_to_ycbcr_y(img_rgb_uint8):
    """BT.601 Y-channel: Y = 16 + (65.481*R + 128.553*G + 24.966*B) / 255"""
    img = img_rgb_uint8.astype(np.float64)
    y = 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0
    return y

def calculate_psnr_y(img1_rgb, img2_rgb, border=BORDER):
    y1 = rgb_to_ycbcr_y(img1_rgb)
    y2 = rgb_to_ycbcr_y(img2_rgb)
    if border > 0:
        y1, y2 = y1[border:-border, border:-border], y2[border:-border, border:-border]
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0: return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim_y(img1_rgb, img2_rgb, border=BORDER):
    y1 = rgb_to_ycbcr_y(img1_rgb)
    y2 = rgb_to_ycbcr_y(img2_rgb)
    if border > 0:
        y1, y2 = y1[border:-border, border:-border], y2[border:-border, border:-border]
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(y1, y2, data_range=235.0 - 16.0)
    except ImportError:
        # Manual SSIM fallback
        C1 = (0.01 * 219.0) ** 2
        C2 = (0.03 * 219.0) ** 2
        mu1 = cv2.GaussianBlur(y1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(y2, (11, 11), 1.5)
        sigma1_sq = cv2.GaussianBlur(y1 ** 2, (11, 11), 1.5) - mu1 ** 2
        sigma2_sq = cv2.GaussianBlur(y2 ** 2, (11, 11), 1.5) - mu2 ** 2
        sigma12 = cv2.GaussianBlur(y1 * y2, (11, 11), 1.5) - mu1 * mu2
        ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(np.mean(ssim_map))

def find_hr_images(data_path):
    hr_paths = []
    hr_dir = os.path.join(data_path, 'HR')
    if os.path.isdir(hr_dir):
        data_path = hr_dir
    for f in sorted(os.listdir(data_path)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            hr_paths.append(os.path.join(data_path, f))
    if not hr_paths:
        for root, dirs, files in os.walk(data_path):
            if 'LR' in root:
                continue
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    hr_paths.append(os.path.join(root, f))
    return sorted(list(set(hr_paths)))


# ─── Run Evaluation ─────────────────────────────────────────────────────────
model.eval()

# Common datasets to find
DATASETS = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']
RESULTS = {}

print("=" * 65)
print("  SSIU-FA MONSTER 48-BLOCK SHORTCUT TO SOTA")
print("=" * 65)
print(f"  Device      : {device}")
print(f"  Blocks      : {NUM_BLOCKS}")
print(f"  Parameters  : {params_k:.1f} K")
print("=" * 65)

for ds_name in DATASETS:
    # Robust search for dataset path
    ds_path = None
    search_dirs = [
        f'/kaggle/input/{ds_name.lower()}/{ds_name}',
        f'/kaggle/input/{ds_name}-hr-lr/{ds_name}',
        f'/kaggle/input/sr-benchmark-datasets/{ds_name}',
        f'/kaggle/input/datasets/chenqizhou/{ds_name.lower()}-hr-lr/{ds_name}',
    ]
    # Check if we can find it by name anywhere in input
    for root, dirs, _ in os.walk('/kaggle/input'):
        if os.path.basename(root).lower() == ds_name.lower():
            ds_path = root
            break
            
    if not ds_path or not os.path.exists(ds_path):
        continue

    hr_paths = find_hr_images(ds_path)
    if not hr_paths: continue

    print(f"\nEvaluating {ds_name} ({len(hr_paths)} images)...")
    psnrs, ssims = [], []
    
    for p in hr_paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None: continue
        h, w, _ = img_bgr.shape
        h, w = h - (h % UPSCALE), w - (w % UPSCALE)
        img_bgr = img_bgr[:h, :w, :]
        hr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        lr_pil = Image.fromarray(hr_rgb).resize((w // UPSCALE, h // UPSCALE), resample=Image.BICUBIC)
        lr_rgb = np.array(lr_pil)

        with torch.no_grad():
            lr_t = torch.from_numpy(lr_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            sr_t = model(lr_t)
            sr_rgb = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        psnrs.append(calculate_psnr_y(sr_rgb, hr_rgb))
        ssims.append(calculate_ssim_y(sr_rgb, hr_rgb))

    avg_psnr, avg_ssim = np.mean(psnrs), np.mean(ssims)
    RESULTS[ds_name] = (avg_psnr, avg_ssim)
    print(f"  > {ds_name}: {avg_psnr:.2f} dB / {avg_ssim:.4f}")

print("\n" + "=" * 65)
print("  🏆 FINAL PERFORMANCE SUMMARY")
print("=" * 65)
print(f"  {'Dataset':<15s} | {'PSNR':>10s} | {'SSIM':>10s}")
print("  " + "-" * 45)
for ds, vals in RESULTS.items():
    print(f"  {ds:<15s} | {vals[0]:>8.2f} dB | {vals[1]:>10.4f}")
print("=" * 65)


# === CELL 5: Download Weights ================================================
# %%
from IPython.display import FileLink
import shutil

# Copy final weights to /kaggle/working for download
shutil.copy("ssiu_fa_24b_final.pth", "/kaggle/working/ssiu_fa_24b_final.pth")

# Also copy the last checkpoint
import glob
ckpts = sorted(glob.glob("ssiu_fa_24b_iter_*.pth"))
if ckpts:
    shutil.copy(ckpts[-1], f"/kaggle/working/{os.path.basename(ckpts[-1])}")

print("✅ Weights saved to /kaggle/working/")
print("   Download them from the Output tab.")
for f in glob.glob("/kaggle/working/ssiu_fa_24b*.pth"):
    print(f"   📦 {os.path.basename(f)}")
