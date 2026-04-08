"""
SSIU-FA Evaluation Script — Kaggle Ready
=========================================
Scale   : x4 ONLY (hardcoded)
Metrics : Y-channel PSNR + Y-channel SSIM
Border  : 4 pixels shaved (standard for x4)
Color   : RGB processing, Y-channel extraction via BT.601
LR Gen  : PIL bicubic (matching training pipeline)

Usage (Kaggle):
    python eval.py --model_path ssiu_fa_x4_final.pth --data_path /kaggle/input/set5/Set5/HR

Usage (Local):
    python eval.py --model_path ssiu_fa_x4_final.pth --data_path datasets/Set5/HR
"""
import torch
import cv2
import numpy as np
import os
import sys
from PIL import Image

from ssiu_improved import ImprovedSSIUNet, SCALE, EMBED_DIM, NUM_BLOCKS


# ─── Fixed Evaluation Configuration ────────────────────────────────────────────
UPSCALE = 4
BORDER = UPSCALE   # Shave border pixels = scale factor
# ────────────────────────────────────────────────────────────────────────────────

# Kaggle Set5 path candidates
KAGGLE_SET5_PATHS = [
    '/kaggle/input/set5-hr-lr/Set5',
    '/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5',
    '/kaggle/input/set5/Set5/HR',
]


# ─── Metric Functions ──────────────────────────────────────────────────────────

def rgb_to_ycbcr_y(img_rgb_uint8):
    """
    Convert RGB uint8 image to Y channel (luminance) using BT.601.

    This is the STANDARD conversion used by all SR papers:
        Y = 16 + (65.481 * R + 128.553 * G + 24.966 * B) / 255

    Args:
        img_rgb_uint8: numpy array [H, W, 3], dtype uint8, channel order RGB
    Returns:
        y_channel: numpy array [H, W], dtype float64, range [16, 235]
    """
    img = img_rgb_uint8.astype(np.float64)
    y = 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0
    return y


def calculate_psnr_y(img1_rgb, img2_rgb, border=BORDER):
    """
    Calculate PSNR on Y-channel with border shaving.

    Args:
        img1_rgb, img2_rgb: [H, W, 3] uint8 RGB images
        border: pixels to shave from each edge
    Returns:
        PSNR in dB (float)
    """
    if img1_rgb.shape != img2_rgb.shape:
        raise ValueError(f"Shape mismatch: {img1_rgb.shape} vs {img2_rgb.shape}")

    y1 = rgb_to_ycbcr_y(img1_rgb)
    y2 = rgb_to_ycbcr_y(img2_rgb)

    if border > 0:
        y1 = y1[border:-border, border:-border]
        y2 = y2[border:-border, border:-border]

    mse = np.mean((y1 - y2) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim_y(img1_rgb, img2_rgb, border=BORDER):
    """
    Calculate SSIM on Y-channel with border shaving.

    Uses skimage if available, falls back to manual implementation.

    Args:
        img1_rgb, img2_rgb: [H, W, 3] uint8 RGB images
        border: pixels to shave from each edge
    Returns:
        SSIM (float, 0-1)
    """
    y1 = rgb_to_ycbcr_y(img1_rgb)
    y2 = rgb_to_ycbcr_y(img2_rgb)

    if border > 0:
        y1 = y1[border:-border, border:-border]
        y2 = y2[border:-border, border:-border]

    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(y1, y2, data_range=235.0 - 16.0)
    except ImportError:
        # Manual SSIM (Wang et al. 2004)
        return _manual_ssim(y1, y2, data_range=235.0 - 16.0)


def _manual_ssim(img1, img2, data_range, k1=0.01, k2=0.03, win_size=11):
    """Fallback SSIM implementation when skimage is not available."""
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    mu1 = cv2.GaussianBlur(img1, (win_size, win_size), 1.5)
    mu2 = cv2.GaussianBlur(img2, (win_size, win_size), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (win_size, win_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (win_size, win_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


# ─── Weight Loading ────────────────────────────────────────────────────────────

def load_model_weights(model, path, device):
    """Load weights with module prefix stripping and flexible checkpoint format."""
    if not os.path.exists(path):
        print(f"  WARNING: Weight file not found: {path}")
        return False
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return True


# ─── Image Discovery ──────────────────────────────────────────────────────────

def find_hr_images(data_path):
    """
    Find HR images in the given path.
    Handles both flat directories and HR/LR-structured benchmark directories.
    """
    hr_paths = []

    # Check if there's an HR subdirectory
    hr_dir = os.path.join(data_path, 'HR')
    if os.path.isdir(hr_dir):
        data_path = hr_dir

    for f in sorted(os.listdir(data_path)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            hr_paths.append(os.path.join(data_path, f))

    # If still empty, walk recursively looking for HR directories
    if not hr_paths:
        for root, dirs, files in os.walk(data_path):
            # Skip LR directories
            if 'LR' in root:
                continue
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    hr_paths.append(os.path.join(root, f))

    return sorted(list(set(hr_paths)))


def auto_detect_set5():
    """Try common Kaggle Set5 paths."""
    for p in KAGGLE_SET5_PATHS:
        if os.path.isdir(p):
            return p
    return None


# ─── Main Evaluation ──────────────────────────────────────────────────────────

def evaluate(model_path, data_path, baseline_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ─── Print Configuration ────────────────────────────────────────────────
    print("=" * 65)
    print("  SSIU-FA EVALUATION — x4 Super-Resolution")
    print("=" * 65)
    print(f"  Device      : {device}")
    print(f"  Scale       : x{UPSCALE}")
    print(f"  Channels    : {EMBED_DIM}")
    print(f"  Blocks      : {NUM_BLOCKS}")
    print(f"  Border shave: {BORDER} px")
    print(f"  Metrics     : Y-channel PSNR (dB) + Y-channel SSIM")
    print(f"  Model       : {model_path}")
    print(f"  Data        : {data_path}")

    # ─── Load Improved Model ────────────────────────────────────────────────
    model = ImprovedSSIUNet(upscale=UPSCALE).to(device)
    params_k = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"  Parameters  : {params_k:.1f} K")

    if not load_model_weights(model, model_path, device):
        print("  WARNING: Cannot load improved model weights. Skipping our model.")
        model = None
    else:
        model.eval()

    # ─── Load Baseline Model (optional) ─────────────────────────────────────
    baseline_model = None
    if baseline_path:
        try:
            from ssiu_official import SSUFSRNet
            from collections import namedtuple
            Args = namedtuple('Args', ['scale', 'n_feats', 'n_blocks', 'colors'])
            baseline_model = SSUFSRNet(Args(4, 64, 9, 3)).to(device)
            if load_model_weights(baseline_model, baseline_path, device):
                baseline_model.eval()
                print(f"  Baseline    : {baseline_path} (loaded)")
            else:
                baseline_model = None
        except ImportError:
            print("  Baseline    : ssiu_official.py not found, skipping comparison")
    print("=" * 65)

    # ─── Find Images ────────────────────────────────────────────────────────
    hr_paths = find_hr_images(data_path)
    if not hr_paths:
        print(f"  ERROR: No HR images found in {data_path}")
        return

    print(f"\n  Found {len(hr_paths)} test images\n")
    print(f"  {'Image':<20s} | {'PSNR (Ours)':>12s} | {'SSIM (Ours)':>12s}", end="")
    if baseline_model:
        print(f" | {'PSNR (Base)':>12s} | {'Diff':>7s}", end="")
    print()
    print("  " + "-" * (58 + (24 if baseline_model else 0)))

    # ─── Per-Image Evaluation ───────────────────────────────────────────────
    psnrs_ours, ssims_ours = [], []
    psnrs_base = []

    for p in hr_paths:
        # Load HR: BGR → RGB
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            print(f"  WARNING: Could not read {p}, skipping")
            continue
        h, w, _ = img_bgr.shape

        # Ensure dimensions are multiples of scale
        h = h - (h % UPSCALE)
        w = w - (w % UPSCALE)
        img_bgr = img_bgr[:h, :w, :]
        hr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Generate LR via PIL bicubic (matching training)
        lr_pil = Image.fromarray(hr_rgb).resize(
            (w // UPSCALE, h // UPSCALE), resample=Image.BICUBIC
        )
        lr_rgb = np.array(lr_pil)

        with torch.no_grad():
            # ─── Our Model (RGB, [0,1]) ─────────────────────────────────
            lr_t = torch.from_numpy(lr_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            
            if model is not None:
                sr_t = model(lr_t)
                sr_rgb = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

                # Ensure output matches HR size
                if sr_rgb.shape[0] != h or sr_rgb.shape[1] != w:
                    sr_rgb = np.array(Image.fromarray(sr_rgb).resize((w, h), resample=Image.BICUBIC))

                psnr_ours = calculate_psnr_y(sr_rgb, hr_rgb)
                ssim_ours = calculate_ssim_y(sr_rgb, hr_rgb)
                psnrs_ours.append(psnr_ours)
                ssims_ours.append(ssim_ours)
            else:
                psnr_ours = 0.0
                ssim_ours = 0.0

            # ─── Baseline Model (RGB, [0,1]) ────────────────────────────
            psnr_base = 0.0
            if baseline_model:
                sr_b_t = baseline_model(lr_t)
                sr_b_rgb = (sr_b_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                if sr_b_rgb.shape[0] != h or sr_b_rgb.shape[1] != w:
                    sr_b_rgb = np.array(Image.fromarray(sr_b_rgb).resize((w, h), resample=Image.BICUBIC))
                psnr_base = calculate_psnr_y(sr_b_rgb, hr_rgb)
                psnrs_base.append(psnr_base)

        name = os.path.basename(p)[:18]
        if model is not None:
            line = f"  {name:<20s} | {psnr_ours:>10.2f} dB | {ssim_ours:>12.4f}"
            if baseline_model:
                diff = psnr_ours - psnr_base
                line += f" | {psnr_base:>10.2f} dB | {diff:>+6.2f} dB"
        else:
            line = f"  {name:<20s} | {'N/A':>10s}    | {'N/A':>12s}"
            if baseline_model:
                line += f" | {psnr_base:>10.2f} dB | {'N/A':>9s}"
                
        print(line)

    # ─── Summary ────────────────────────────────────────────────────────────
    print("  " + "-" * (58 + (24 if baseline_model else 0)))
    
    # Safely compute averages
    avg_psnr, avg_ssim = 0.0, 0.0
    if model is not None and psnrs_ours:
        avg_psnr = np.mean(psnrs_ours)
        avg_ssim = np.mean(ssims_ours)

    summary = f"  {'AVERAGE':<20s} | "
    if model is not None:
        summary += f"{avg_psnr:>10.2f} dB | {avg_ssim:>12.4f}"
    else:
        summary += f"{'N/A':>10s}    | {'N/A':>12s}"
        
    if baseline_model and psnrs_base:
        avg_base = np.mean(psnrs_base)
        if model is not None:
            summary += f" | {avg_base:>10.2f} dB | {avg_psnr - avg_base:>+6.2f} dB"
        else:
            summary += f" | {avg_base:>10.2f} dB | {'N/A':>9s}"
            
    print(summary)

    # Paper benchmark reference
    print(f"\n  Paper baseline (Set5 x4): 32.64 dB PSNR / 0.9002 SSIM")
    print(f"  Our result              : {avg_psnr:.2f} dB PSNR / {avg_ssim:.4f} SSIM")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SSIU-FA Evaluation (x4 only)")
    parser.add_argument('--model_path', type=str, default='ssiu_fa_x4_final.pth',
                        help="Path to improved model weights")
    parser.add_argument('--data_path', type=str, default=None,
                        help="Path to test dataset (e.g., Set5/HR)")
    parser.add_argument('--baseline_path', type=str, default=None,
                        help="Path to baseline weights for comparison (optional)")
    args = parser.parse_args()

    # Auto-detect
    data_path = args.data_path or auto_detect_set5()
    if data_path is None:
        print("ERROR: No data_path provided and auto-detect failed.")
        print("Usage: python eval.py --model_path model.pth --data_path /path/to/Set5/HR")
        sys.exit(1)

    evaluate(
        model_path=args.model_path,
        data_path=data_path,
        baseline_path=args.baseline_path
    )
