import torch
import cv2
import numpy as np
import os

# Import architectures
from ssiu_improved import ImprovedSSIUNet
try:
    from ssiu_official import SSUFSRNet
    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False

def calculate_psnr(img1, img2, border=4):
    if img1.shape != img2.shape: return 0
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    if border > 0:
        y1, y2 = y1[border:-border, border:-border], y2[border:-border, border:-border]
    mse = np.mean((y1 - y2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def load_weights(model, path):
    if not os.path.exists(path): return False
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return True

def validate(model_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Models
    model_i = ImprovedSSIUNet(upscale=4).to(device)
    load_weights(model_i, model_path)
    model_i.eval()
    
    model_b = None
    if BASELINE_AVAILABLE:
        from collections import namedtuple
        Args = namedtuple('Args', ['scale', 'n_feats', 'n_blocks', 'colors'])
        model_b = SSUFSRNet(Args(4, 64, 10, 3)).to(device)
        load_weights(model_b, "pretrain_model/model_x4_290.pt")
        model_b.eval()

    # 2. Find Images
    hr_dir = data_path
    if not os.path.exists(hr_dir): return
    
    # Simple recursive search for images
    hr_paths = []
    for root, _, files in os.walk(hr_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.bmp')):
                hr_paths.append(os.path.join(root, f))
    hr_paths = sorted(hr_paths)
    
    if not hr_paths:
        print(f"Error: No images found in {hr_dir}")
        return

    # DIV2K Mean
    mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1).to(device)
    
    print(f"🎯 Benchmark Start | Images: {len(hr_paths)} | Device: {device}")
    print("-" * 65)
    
    psnrs_i, psnrs_b = [], []
    for p in hr_paths:
        img = cv2.imread(p)
        h, w, _ = img.shape
        img = img[:h-(h%4), :w-(w%4), :]
        hr_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        lr_bgr = cv2.resize(img, (w//4, h//4), interpolation=cv2.INTER_CUBIC)
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            # Improved (RGB, 0-1)
            t_i = torch.from_numpy(lr_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            sr_i = (model_i(t_i).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pi = calculate_psnr(sr_i, hr_rgb)
            psnrs_i.append(pi)
            
            # Baseline (BGR, Mean-Shift, 0-1)
            pb = 0
            if model_b:
                t_b = (torch.from_numpy(lr_bgr.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0) - mean
                sr_b = ((model_b(t_b) + mean).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                pb = calculate_psnr(cv2.cvtColor(sr_b, cv2.COLOR_BGR2RGB), hr_rgb)
                psnrs_b.append(pb)

        print(f"{os.path.basename(p)[:15]:<15} | Base: {pb:.2f} dB | Ours: {pi:.2f} dB")

    print(f"\n✅ SUMMARY | Baseline: {np.mean(psnrs_b):.2f} dB | Ours: {np.mean(psnrs_i):.2f} dB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="ssiu-div2k.pth")
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    validate(args.model_path, args.data_path)
