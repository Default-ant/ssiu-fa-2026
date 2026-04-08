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
    # Correct Academic BT.601 Y-channel conversion
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    if border > 0:
        y1, y2 = y1[border:-border, border:-border], y2[border:-border, border:-border]
    mse = np.mean((y1 - y2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def test_baseline(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Official Specs (SSIU 2025)
    from collections import namedtuple
    Args = namedtuple('Args', ['scale', 'n_feats', 'n_blocks', 'colors'])
    model = SSUFSRNet(Args(scale=4, n_feats=64, n_blocks=10, colors=3)).to(device)
    
    # 2. Load Weights
    weight_path = "pretrain_model/model_x4_290.pt"
    if not os.path.exists(weight_path):
        print(f"Error: {weight_path} not found.")
        return
        
    ckpt = torch.load(weight_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    
    # 3. Find HR Images (Set5 focus)
    hr_paths = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.bmp')):
                hr_paths.append(os.path.join(root, f))
    hr_paths = sorted(list(set(hr_paths)))[:5] 
    
    if not hr_paths:
        print("No images found.")
        return

    # DIV2K Mean (Standard 0-1 range)
    mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1).to(device)
    
    print(f"🚀 Official Baseline Verification (0.0-1.0 Range Mode)")
    print("-" * 50)
    
    psnrs = []
    for p in hr_paths:
        img_bgr = cv2.imread(p)
        h, w, _ = img_bgr.shape
        img_bgr = img_bgr[:h-(h%4), :w-(w%4), :]
        
        # High-Quality Alignment: Use PIL for Bicubic (Matches MATLAB imresize closer than CV2)
        lr_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).resize((w//4, h//4), resample=Image.BICUBIC)
        lr_rgb = np.array(lr_pil)
        
        # Standard Normalization: [0, 1] + Mean Subtraction
        t = torch.from_numpy(lr_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        t = t - mean
        
        with torch.no_grad():
            sr_t = model(t) + mean
            sr = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            # HR for comparison
            hr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            psnr = calculate_psnr(sr, hr_rgb)
            psnrs.append(psnr)
            print(f"{os.path.basename(p):<15} | PSNR: {psnr:.2f} dB")

    print("-" * 50)
    print(f"✅ Final Average Baseline: {np.mean(psnrs):.2f} dB")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5"
    test_baseline(path)
