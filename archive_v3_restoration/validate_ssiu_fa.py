"""
SSIU-FA Validation Script
==========================
Validates SSUFSRNet_FA on standard SR benchmarks.
Reports PSNR (RGB, full image) - same protocol as the original SSIU 2025 paper.
"""
import torch
import cv2
import numpy as np
import os
import argparse
from ssiu_fa_network import SSUFSRNet_FA


class SimpleArgs:
    def __init__(self, scale=4):
        self.scale = scale
        self.n_feats = 64
        self.n_blocks = 9
        self.colors = 3


def rgb_to_ycbcr(img):
    """
    Standard RGB to YCbCr conversion (BT.601) as used in research papers.
    Returns the Y-channel (luminance) normalized to [0, 255].
    """
    img = img.astype(np.float64)
    # BT.601 coefficients
    y = 16.0 + (65.481 * img[:,:,0] + 128.553 * img[:,:,1] + 24.966 * img[:,:,2]) / 255.0
    return y


def calculate_psnr(img1, img2, border=0):
    """Calculate PSNR on Y-channel with boundary shaving."""
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def validate(model_path, data_path, scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = SimpleArgs(scale=scale)
    model = SSUFSRNet_FA(args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if not os.path.exists(data_path):
        print(f"Error: Test directory {data_path} not found!")
        return None

    hr_paths = sorted([
        os.path.join(data_path, f) for f in os.listdir(data_path)
        if f.lower().endswith(('.png', '.jpg', '.bmp'))
    ])

    psnrs = []
    dataset_name = os.path.basename(data_path.rstrip('/'))
    print(f"\n--- SSIU-FA x{scale} SCIENTIFIC VALIDATION (Y-channel) on {dataset_name} ---")

    for hr_path in hr_paths:
        hr = cv2.imread(hr_path)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        h, w, _ = hr.shape

        # Pad for window size consistency
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        hr_pad = np.pad(hr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect') if (pad_h or pad_w) else hr
        
        # Downsample (OpenCV Bicubic is the closest to MATLAB outside of MATLAB itself)
        lr_w = hr_pad.shape[1] // scale
        lr_h = hr_pad.shape[0] // scale
        lr = cv2.resize(hr_pad, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            sr_t = model(lr_t.to(device))
            sr = sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            sr = (sr * 255.0).clip(0, 255)

        # Crop to matching size
        target_h, target_w = h, w
        sr = sr[:target_h, :target_w, :]
        hr_eval = hr[:target_h, :target_w, :]

        # Convert both to Y-channel for scientific comparison
        sr_y = rgb_to_ycbcr(sr)
        hr_y = rgb_to_ycbcr(hr_eval)

        # Calculate PSNR with boundary shaving (standard research border = scale)
        p = calculate_psnr(hr_y, sr_y, border=scale)
        psnrs.append(p)
        print(f"  {os.path.basename(hr_path):30s} | PSNR: {p:.2f} dB (Y)")

    avg = np.mean(psnrs)

    # SSIU 2025 paper baselines (Table 1) - THESE ARE Y-CHANNEL PSNRS
    sota = {
        2: {'set5': 38.31, 'set14': 34.20, 'bsd100': 32.43, 'urban100': 33.25, 'manga109': 39.64},
        3: {'set5': 34.79, 'set14': 30.71, 'bsd100': 29.35, 'urban100': 29.08, 'manga109': 34.73},
        4: {'set5': 32.64, 'set14': 28.96, 'bsd100': 27.82, 'urban100': 26.83, 'manga109': 31.60},
    }
    ds = 'set5'
    for key in sota[scale]:
        if key in data_path.lower():
            ds = key
            break

    baseline = sota[scale][ds]
    print(f"\n{'='*45}")
    print(f"📊 SSIU-FA x{scale} | {dataset_name}")
    print(f"   Average PSNR : {avg:.2f} dB")
    print(f"   SSIU 2025    : {baseline:.2f} dB  (baseline)")
    delta = avg - baseline
    if delta >= 0:
        print(f"   STATUS       : 🏆 BEAT BASELINE by +{delta:.2f} dB")
    else:
        print(f"   STATUS       : Gap: {delta:.2f} dB to baseline")
    print(f"{'='*45}\n")
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path',  type=str, required=True)
    parser.add_argument('--scale',      type=int, default=4)
    args = parser.parse_args()
    validate(args.model_path, args.data_path, args.scale)
