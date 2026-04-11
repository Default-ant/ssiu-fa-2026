"""
SSIU-FA Fine-Tuning Script
===========================
Loads original SSIU pretrained weights and fine-tunes the SALK-enhanced layers.
Expected duration: ~30-45 minutes for 10,000 iterations on Kaggle GPU.
Expected result: > 32.64 dB on Set5 x4 (beating original SSIU 2025 baseline)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import random
import argparse
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ssiu_fa_network import SSUFSRNet_FA


class SimpleArgs:
    def __init__(self, scale=4):
        self.scale = scale
        self.n_feats = 64
        self.n_blocks = 9
        self.colors = 3


class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, hr_dir, scale=4, patch_size=48, num_patches=16):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size  # HR patch size
        self.num_patches = num_patches
        self.hr_paths = sorted([
            os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.bmp'))
        ])
        print(f"Loaded {len(self.hr_paths)} training images.")

    def __len__(self):
        return len(self.hr_paths) * self.num_patches

    def __getitem__(self, idx):
        img_idx = idx // self.num_patches
        hr = cv2.imread(self.hr_paths[img_idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        h, w, _ = hr.shape
        ps = self.patch_size
        lps = ps // self.scale

        # Random crop
        x = random.randint(0, max(h - ps, 0))
        y = random.randint(0, max(w - ps, 0))
        hr_patch = hr[x:x+ps, y:y+ps, :]
        lr_patch = cv2.resize(hr_patch, (lps, lps), interpolation=cv2.INTER_CUBIC)

        # Augmentations
        if random.random() > 0.5:
            hr_patch = np.fliplr(hr_patch).copy()
            lr_patch = np.fliplr(lr_patch).copy()
        if random.random() > 0.5:
            hr_patch = np.flipud(hr_patch).copy()
            lr_patch = np.flipud(lr_patch).copy()

        hr_t = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0
        lr_t = torch.from_numpy(lr_patch).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target)**2 + eps**2))


def fine_tune(pretrained_path, data_path, scale=4, iterations=10000, lr=2e-4, save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build model
    args = SimpleArgs(scale=scale)
    model = SSUFSRNet_FA(args).to(device)

    # Load pretrained weights (partial - SALK layers get random init)
    print(f"\nLoading pretrained SSIU weights from: {pretrained_path}")
    loaded, skipped = model.load_pretrained(pretrained_path, device=device)

    # Only fine-tune SALK layers and quickly adapt the rest
    # Strategy: lower LR for pretrained layers, higher LR for new SALK layers
    salk_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'salk' in name or 'project_in2_salk' in name:
            salk_params.append(param)
        else:
            base_params.append(param)

    optimizer = Adam([
        {'params': salk_params, 'lr': lr},       # New SALK layers: full LR
        {'params': base_params, 'lr': lr * 0.1},  # Pretrained layers: 10x lower LR
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-6)

    # Dataset
    dataset = DIV2KDataset(data_path, scale=scale, patch_size=48*scale//4, num_patches=32)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    model.train()
    data_iter = iter(loader)
    total_loss = 0

    if save_path is None:
        save_path = f"ssiu_fa_x{scale}_finetuned.pth"

    print(f"\n{'='*50}")
    print(f"🔥 FINE-TUNING SSIU-FA (x{scale}) for {iterations} iterations")
    print(f"   New SALK layers: LR={lr:.5f}")
    print(f"   Pretrained layers: LR={lr*0.1:.5f}")
    print(f"{'='*50}\n")

    for i in range(1, iterations + 1):
        try:
            lr_imgs, hr_imgs = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            lr_imgs, hr_imgs = next(data_iter)

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        optimizer.zero_grad()
        sr = model(lr_imgs)
        loss = charbonnier_loss(sr, hr_imgs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if i % 500 == 0 or i == 1:
            avg_loss = total_loss / (i if i < 500 else 500)
            total_loss = 0
            print(f"⚡ Iter {i:5d}/{iterations} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if i % 5000 == 0:
            ckpt_path = save_path.replace('.pth', f'_iter{i}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Fine-tuning complete! Model saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, required=True, help='Path to original SSIU .pt file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to DIV2K HR training images')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    fine_tune(
        pretrained_path=args.pretrained,
        data_path=args.data_path,
        scale=args.scale,
        iterations=args.iterations,
        lr=args.lr,
        save_path=args.save_path
    )
