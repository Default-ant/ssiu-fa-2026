import torch
import cv2
import numpy as np
import os
import sys
from PIL import Image

# Import the official architecture
try:
    from ssiu_official import SSUFSRNet
except ImportError:
    print("Error: ssiu_official.py not found in root.")
    sys.exit(1)

def calculate_psnr(img1, img2, border=4):
    if img1.shape != img2.shape: return 0
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    # Correct Academic BT.601 Y-channel conversion (Standard for SOTA)
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    if border > 0:
        y1, y2 = y1[border:-border, border:-border], y2[border:-border, border:-border]
    mse = np.mean((y1 - y2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def test_baseline(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. VERIFIED SPECS (Based on weight inspection)
    from collections import namedtuple
    Args = namedtuple('Args', ['scale', 'n_feats', 'n_blocks', 'colors'])
    
    # We found 9 blocks (0-8) in the weights!
    model = SSUFSRNet(Args(scale=4, n_feats=64, n_blocks=9, colors=3)).to(device)
    
    # 2. Load Weights
    weight_path = "pretrain_model/model_x4_290.pt"
    if not os.path.exists(weight_path):
        print(f"Error: {weight_path} not found.")
        return
        
    ckpt = torch.load(weight_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    # Strip 'module.' and ensure strict compatibility
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    
    # 3. Find HR Images
    hr_paths = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg')) and "baby" in f.lower():
                hr_paths.append(os.path.join(root, f))
    
    if not hr_paths:
        print("No baby.png found for testing.")
        return

    print(f"🚀 Weight-Verified Baseline (n_blocks=9, n_feats=64)")
    print("-" * 50)
    
    psnrs = []
    for p in hr_paths:
        img_bgr = cv2.imread(p)
        h, w, _ = img_bgr.shape
        img_bgr = img_bgr[:h-(h%4), :w-(w%4), :]
        hr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # SOTA Standard: PIL Bicubic
        lr_pil = Image.fromarray(hr_rgb).resize((w//4, h//4), resample=Image.BICUBIC)
        lr_rgb = np.array(lr_pil)
        
        # Testing [0, 1] range ONLY (Proven most likely by sweep)
        t = torch.from_numpy(lr_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            sr_t = model(t)
            sr = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            psnr = calculate_psnr(sr, hr_rgb)
            psnrs.append(psnr)
            print(f"{os.path.basename(p):<15} | PSNR: {psnr:.2f} dB")

    print("-" * 50)
    print(f"✅ Final Average Baseline: {np.mean(psnrs):.2f} dB")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5"
    test_baseline(path)
