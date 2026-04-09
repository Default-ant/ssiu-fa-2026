"""
SSIU-FA Training Script — Kaggle Ready
=======================================
Scale   : x4 ONLY (hardcoded)
Dataset : DIV2K HR → bicubic downsampled LR
Color   : RGB (loaded via cv2 BGR → converted to RGB)
LR Gen  : PIL bicubic (MATLAB-like, academic standard)
Loss    : Charbonnier + 0.05 * FFT frequency loss
Optim   : Adam + Cosine Annealing
AMP     : Enabled for Kaggle T4/P100

Usage (Kaggle):
    python train.py --data_path /kaggle/input/div2k-dataset/DIV2K_train_HR
    python train.py --data_path /kaggle/input/div2k-dataset/DIV2K_train_HR --resume checkpoint.pth

Usage (Local):
    python train.py --data_path /path/to/DIV2K_train_HR --iterations 100
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import sys
from PIL import Image

from ssiu_improved import ImprovedSSIUNet, SCALE, EMBED_DIM, NUM_BLOCKS


# ─── Fixed Training Configuration ──────────────────────────────────────────────
UPSCALE = 4
PATCH_SIZE_LR = 64          # LR patch size
PATCH_SIZE_HR = PATCH_SIZE_LR * UPSCALE   # 256, matches baseline config
BATCH_SIZE = 32
DEFAULT_ITERATIONS = 50000
LEARNING_RATE = 1e-3
ETA_MIN = 1e-6
FFT_LOSS_WEIGHT = 0.05
# ────────────────────────────────────────────────────────────────────────────────

# Common Kaggle DIV2K paths (auto-detect)
KAGGLE_PATHS = [
    '/kaggle/input/div2k-dataset/DIV2K_train_HR',
    '/kaggle/input/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR',
    '/kaggle/input/datasets/harshraone/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR',
    '/kaggle/input/div2k-high-resolution-images/DIV2K_train_HR',
]


class DIV2KDataset(Dataset):
    """
    DIV2K training dataset for x4 super-resolution.

    Pipeline:
      1. Load HR image via cv2 (BGR)
      2. Convert to RGB
      3. Random crop to 256×256 HR patch
      4. Random augmentation (flip/rotate)
      5. Generate LR via PIL bicubic → 64×64
      6. Normalize to [0, 1] float
    """
    def __init__(self, data_path):
        super().__init__()
        self.upscale = UPSCALE
        self.patch_size_lr = PATCH_SIZE_LR
        self.patch_size_hr = PATCH_SIZE_HR

        self.file_list = sorted([
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        if len(self.file_list) == 0:
            raise ValueError(
                f"No images found in {data_path}.\n"
                "Check path — you may need to go one level deeper."
            )
        print(f"  Dataset     : {len(self.file_list)} images from {data_path}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. Load BGR → RGB
        img_bgr = cv2.imread(self.file_list[idx])
        hr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = hr.shape

        # 2. Random crop (HR patch)
        x = np.random.randint(0, max(w - self.patch_size_hr, 1))
        y = np.random.randint(0, max(h - self.patch_size_hr, 1))
        hr_crop = hr[y:y + self.patch_size_hr, x:x + self.patch_size_hr]

        # 3. Random augmentation: flips + rotations (8 orientations)
        aug = np.random.randint(0, 8)
        if aug == 1: hr_crop = np.flipud(hr_crop)
        elif aug == 2: hr_crop = np.fliplr(hr_crop)
        elif aug == 3: hr_crop = np.rot90(hr_crop, 1)
        elif aug == 4: hr_crop = np.rot90(hr_crop, 2)
        elif aug == 5: hr_crop = np.rot90(hr_crop, 3)
        elif aug == 6: hr_crop = np.flipud(np.rot90(hr_crop, 1))
        elif aug == 7: hr_crop = np.fliplr(np.rot90(hr_crop, 1))

        # 4. Generate LR via PIL bicubic (academic standard, MATLAB-like)
        hr_pil = Image.fromarray(hr_crop)
        lr_pil = hr_pil.resize(
            (self.patch_size_lr, self.patch_size_lr),
            resample=Image.BICUBIC
        )
        lr_crop = np.array(lr_pil)

        # 5. To tensor [0, 1]
        hr_t = torch.from_numpy(hr_crop.copy()).permute(2, 0, 1).float() / 255.0
        lr_t = torch.from_numpy(lr_crop.copy()).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t


class CharbonnierLoss(nn.Module):
    """Charbonnier (L1 smooth) loss — standard for SR training."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def frequency_loss(sr, hr):
    """FFT-based frequency loss for high-frequency detail recovery."""
    sr_fft = torch.fft.rfft2(sr, norm='ortho')
    hr_fft = torch.fft.rfft2(hr, norm='ortho')
    return torch.mean(torch.abs(sr_fft - hr_fft))


def auto_detect_data_path():
    """Try common Kaggle DIV2K dataset paths."""
    for p in KAGGLE_PATHS:
        if os.path.isdir(p):
            return p
    return None


def train(data_path, iterations=DEFAULT_ITERATIONS, resume_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ─── Print Configuration ────────────────────────────────────────────────
    print("=" * 60)
    print("  SSIU-FA TRAINING — x4 Super-Resolution")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Scale       : x{UPSCALE}")
    print(f"  Channels    : {EMBED_DIM}")
    print(f"  Blocks      : {NUM_BLOCKS}")
    print(f"  Patch (LR)  : {PATCH_SIZE_LR}×{PATCH_SIZE_LR}")
    print(f"  Patch (HR)  : {PATCH_SIZE_HR}×{PATCH_SIZE_HR}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Iterations  : {iterations}")
    print(f"  LR          : {LEARNING_RATE} → {ETA_MIN} (cosine)")
    print(f"  Loss        : Charbonnier + {FFT_LOSS_WEIGHT}×FFT")
    print(f"  AMP         : Enabled")

    # ─── Model ──────────────────────────────────────────────────────────────
    model = ImprovedSSIUNet(upscale=UPSCALE, embed_dim=EMBED_DIM,
                            num_blocks=NUM_BLOCKS).to(device)
    params_k = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"  Parameters  : {params_k:.1f} K")

    # ─── Resume ─────────────────────────────────────────────────────────────
    start_iter = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        
        # Determine iteration from checkpoint or filename
        if isinstance(ckpt, dict) and 'iteration' in ckpt:
            start_iter = ckpt['iteration']
            sd = ckpt['model_state_dict']
        else:
            sd = ckpt if isinstance(ckpt, dict) and 'conv_in.weight' in ckpt else ckpt.get('model_state_dict', ckpt)
            # Try to parse iteration from filename (e.g., ssiu_fa_x4_iter_30000.pth)
            import re
            match = re.search(r'iter_(\d+)', os.path.basename(resume_path))
            if match:
                start_iter = int(match.group(1))
        
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
        
        # Optional: load optimizer if it exists
        if isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
            optimizer_temp = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            optimizer_temp.load_state_dict(ckpt['optimizer_state_dict'])
            # We will re-init optimizer properly below but this confirms it's possible
        
        print(f"  Resumed from: {resume_path} (Iteration {start_iter})")

    # ─── Dataset ────────────────────────────────────────────────────────────
    dataset = DIV2KDataset(data_path)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    print("=" * 60)

    # ─── Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    # CRITICAL: Set initial_lr for scheduler if resuming
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', LEARNING_RATE)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, 
                                                     eta_min=ETA_MIN, last_epoch=start_iter if start_iter > 0 else -1)
    
    # If resuming, we need to load optimizer state if available
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    criterion = CharbonnierLoss()
    scaler = torch.amp.GradScaler('cuda')

    # ─── Training Loop ──────────────────────────────────────────────────────
    model.train()
    loader_iter = iter(dataloader)

    for i in range(start_iter + 1, iterations + 1):
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
            print(f"  [{i:6d}/{iterations}] "
                  f"Loss: {loss.item():.5f} "
                  f"(Char: {loss_char.item():.5f} | FFT: {loss_fft.item():.5f}) "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if i % 5000 == 0 or i == iterations:
            ckpt_path = f"ssiu_fa_x4_iter_{i}.pth"
            save_data = {
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(save_data, ckpt_path)
            print(f"  >>> Checkpoint saved: {ckpt_path}")

    # ─── Final Save ─────────────────────────────────────────────────────────
    final_path = "ssiu_fa_x4_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n  TRAINING COMPLETE. Final weights: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SSIU-FA Training (x4 only)")
    parser.add_argument('--data_path', type=str, default=None,
                        help="Path to DIV2K_train_HR directory")
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint .pth to resume from")
    args = parser.parse_args()

    # Auto-detect Kaggle path if not provided
    data_path = args.data_path or auto_detect_data_path()
    if data_path is None:
        print("ERROR: No data_path provided and auto-detect failed.")
        print("Usage: python train.py --data_path /path/to/DIV2K_train_HR")
        sys.exit(1)

    train(data_path=data_path, iterations=args.iterations, resume_path=args.resume)
