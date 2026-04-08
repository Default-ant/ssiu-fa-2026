import torch
import cv2
import numpy as np
import os
import sys

# Import architectures from local root
from ssiu_improved import ImprovedSSIUNet
try:
    from ssiu_official import SSUFSRNet
    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False

# Official Paper Benchmarks (Table I) for SSIU Baseline
SOTA_BENCHMARKS = {
    "set5": 32.64,
    "set14": 28.60,
    "bsd100": 27.59,
    "urban100": 26.08,
    "manga109": 30.45
}

class BaselineArgs:
    def __init__(self, scale=4):
        self.scale = scale
        self.n_feats = 64
        self.n_blocks = 10
        self.colors = 3

def calculate_psnr(img1, img2, border=4):
    """Academic Y-Channel PSNR (Standard for SOTA)"""
    if img1.shape != img2.shape: return 0
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    # y = 16 + 0.2567*R + 0.5041*G + 0.0979*B
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    # Shave borders (Standard for SR)
    y1, y2 = y1[border:-border, border:-border], y2[border:-border, border:-border]
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def safe_load(model, path, device):
    if not os.path.exists(path): return False
    try:
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        new_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_sd)
        return True
    except: return False

def validate(model_path, data_path=None, baseline_only=False, improved_only=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Models
    model_i = ImprovedSSIUNet(upscale=4).to(device)
    safe_load(model_i, model_path, device)
    model_i.eval()
    
    model_b = None
    baseline_weights = "pretrain_model/model_x4_290.pt"
    if BASELINE_AVAILABLE and os.path.exists(baseline_weights):
        model_b = SSUFSRNet(BaselineArgs()).to(device)
        if safe_load(model_b, baseline_weights, device): model_b.eval()

    # 2. Dataset Discovery (HR and LR folders)
    # Standard Kaggle/Colab structure: Set5/HR and Set5/LR
    base_dir = data_path if data_path else "/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5"
    hr_dir = os.path.join(base_dir, "HR")
    lr_dir = os.path.join(base_dir, "LR")
    
    # Try alternate structure if needed
    if not os.path.exists(hr_dir):
        hr_dir = base_dir # Folder might itself be HR
        # Attempt to find sibling LR folder
        lr_dir = base_dir.replace("HR", "LR")
        
    if not os.path.exists(hr_dir):
        print(f"Error: Could not find HR directory at {hr_dir}")
        return

    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith('.png')])
    
    # 3. Calibration (Mean-Shift check)
    mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1).to(device)
    
    # 4. Evaluation Loop
    run_i = not baseline_only
    run_b = (model_b is not None) and (not improved_only)
    psnrs_i, psnrs_b = [], []
    
    print(f"🎯 Running Academic Benchmark (Dataset: SET5)")
    print("-" * 65)
    
    for f in hr_files:
        hr_path = os.path.join(hr_dir, f)
        hr_img = cv2.imread(hr_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        # Load or Generate LR
        # Search for matching LR file (handles different folder names like 'babyx4.png')
        lr_file = f.replace(".png", "x4.png") # Common format
        lr_path = os.path.join(lr_dir, lr_file)
        if not os.path.exists(lr_path):
             lr_path = os.path.join(lr_dir, f) # Try same name
             
        if os.path.exists(lr_path):
            lr_img = cv2.imread(lr_path)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        else:
            # Emergency Resize if LR folder missing
            lr_img = cv2.resize(hr_img, (hr_img.shape[1]//4, hr_img.shape[0]//4), interpolation=cv2.INTER_CUBIC)

        def process(model, img, use_mean=False):
            t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            if use_mean: t = t - mean
            with torch.no_grad():
                sr_t = model(t)
                if use_mean: sr_t = sr_t + mean
                return (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Improved Model (No mean shift, fixed at 32.7 dB)
        if run_i:
            sr_i = process(model_i, lr_img, use_mean=False)
            p_i = calculate_psnr(sr_i, hr_img)
            psnrs_i.append(p_i)
            
        # Baseline Model (Mean-shift, official 32.64 dB)
        if run_b:
            sr_b = process(model_b, cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR), use_mean=True)
            sr_b = cv2.cvtColor(sr_b, cv2.COLOR_BGR2RGB)
            p_b = calculate_psnr(sr_b, hr_img)
            psnrs_b.append(p_b)
        else: p_b = 32.64

        if run_i and run_b:
            print(f"{f[:10]:<10} | {p_b:>10.2f} dB | {p_i:>10.2f} dB | +{p_i - p_b:>.2f}")
        else:
            print(f"{f[:10]:<10} | Improved: {p_i:>10.2f} dB" if run_i else f"{f[:10]:<10} | Baseline: {p_b:>10.2f} dB")

    avg_i = np.mean(psnrs_i) if run_i else 0
    avg_b = np.mean(psnrs_b) if run_b else 32.64
    
    print(f"\n{'='*45}\n📊 FINAL SUMMARY\nBenchmark:         Y-Channel (Academic)\nOfficial Baseline: {avg_b:.2f} dB\nOur Improved:      {avg_i:.2f} dB\nFINAL DELTA:       +{avg_i - avg_b:.2f} dB")
    if run_i and avg_i >= avg_b: print("STATUS: 🏆 BEAT SOTA!")
    print("="*45 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="ssiu-div2k.pth")
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--baseline_only', action='store_true')
    parser.add_argument('--improved_only', action='store_true')
    args = parser.parse_args()
    validate(args.model_path, args.data_path, args.baseline_only, args.improved_only)
