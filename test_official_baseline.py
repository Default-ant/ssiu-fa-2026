import torch
import cv2
import numpy as np
import os
import sys

# Import the official architecture
try:
    from ssiu_official import SSUFSRNet
except ImportError:
    print("Error: ssiu_official.py not found in root.")
    sys.exit(1)

def calculate_psnr(img1, img2, border=4):
    if img1.shape != img2.shape: return 0
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    # IEEE BT.601 Y-channel conversion
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
    # Search for Set5 specifically if it's in a subfolder
    for root, _, files in os.walk(data_path):
        if "Set5" in root or "set5" in root:
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.bmp')) and "LR" not in root:
                    hr_paths.append(os.path.join(root, f))
    
    # Fallback to general search if no Set5 folder found
    if not hr_paths:
        for root, _, files in os.walk(data_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.bmp')):
                    hr_paths.append(os.path.join(root, f))
                    
    hr_paths = sorted(list(set(hr_paths)))[:5] # Take first 5 for Set5 check
    
    if not hr_paths:
        print("No images found.")
        return

    # DIV2K Mean (Industry standard for SR)
    # Note: We use the 255-range mean as per BasicSR/KAIR standard
    mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1).to(device) * 255.0
    
    print(f"🚀 Official Baseline Verification (Target: 32.64 dB)")
    print(f"Images: {[os.path.basename(p) for p in hr_paths]}")
    print("-" * 50)
    
    psnrs = []
    for p in hr_paths:
        img_bgr = cv2.imread(p)
        h, w, _ = img_bgr.shape
        img_bgr = img_bgr[:h-(h%4), :w-(w%4), :]
        
        # Official Pre-pro: BGR -> Tensor [0, 255] -> Mean Subtraction
        lr_bgr = cv2.resize(img_bgr, (w//4, h//4), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(lr_bgr.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)
        t = t - mean # EDSR/RCAN/SSIU Standard
        
        with torch.no_grad():
            sr_t = model(t) + mean
            sr = sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
            
            # Move back to RGB for Y-channel PSNR (Academic Standard)
            sr_rgb = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
            hr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            psnr = calculate_psnr(sr_rgb, hr_rgb)
            psnrs.append(psnr)
            print(f"{os.path.basename(p):<15} | PSNR: {psnr:.2f} dB")

    print("-" * 50)
    print(f"✅ Final Average Baseline: {np.mean(psnrs):.2f} dB")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5"
    test_baseline(path)
