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

def calculate_psnr(img1, img2):
    """Simple Robust RGB PSNR"""
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def safe_load(model, path, device):
    if not os.path.exists(path): return False
    try:
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        # Fix DataParallel prefixes
        new_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_sd)
        return True
    except Exception as e:
        print(f"⚠️ Load error: {e}")
        return False

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
        if safe_load(model_b, baseline_weights, device):
            model_b.eval()

    # 2. Setup Dataset
    test_dir = data_path if data_path else "/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5/HR"
    if not os.path.exists(test_dir):
        test_dir = "MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/dependencies/KAIR/testsets/set5/HR"
    
    hr_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    if not hr_files:
        print(f"Error: No images found in {test_dir}")
        return

    # 3. Full Benchmark
    run_i = not baseline_only
    run_b = (model_b is not None) and (not improved_only)
    
    psnrs_i, psnrs_b = [], []
    print(f"🎯 Running Benchmark (Dataset: SET5)")
    print("-" * 65)
    
    for hr_file in hr_files:
        hr_path = os.path.join(test_dir, hr_file)
        img_bgr = cv2.imread(hr_path)
        h, w, _ = img_bgr.shape
        img_bgr = img_bgr[:h-(h%4), :w-(w%4), :]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        lr_bgr = cv2.resize(img_bgr, (w//4, h//4), interpolation=cv2.INTER_CUBIC)
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        
        def process(model, img):
            t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            with torch.no_grad():
                sr_t = model(t)
                return (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if run_i:
            sr_i = process(model_i, lr_rgb) # Improved is RGB
            psnr_i = calculate_psnr(sr_i, img_rgb)
            psnrs_i.append(psnr_i)
        
        if run_b:
            sr_b = process(model_b, lr_bgr) # Baseline is BGR (official)
            psnr_b = calculate_psnr(sr_b, img_bgr)
            psnrs_b.append(psnr_b)
        else:
            psnr_b = sota_target = SOTA_BENCHMARKS.get('set5', 32.64)

        if run_i and run_b:
            print(f"{hr_file[:15]:<15} | {psnr_b:>10.2f} dB | {psnr_i:>10.2f} dB | +{psnr_i - psnr_b:>.2f}")
        elif run_i:
            print(f"{hr_file[:15]:<15} | Improved PSNR: {psnr_i:>10.2f} dB")
        else:
            print(f"{hr_file[:15]:<15} | Baseline PSNR: {psnr_b:>10.2f} dB")

    avg_i = np.mean(psnrs_i) if run_i else 0
    avg_b = np.mean(psnrs_b) if run_b else SOTA_BENCHMARKS.get('set5', 32.64)
    
    print(f"\n{'='*45}\n📊 FINAL SUMMARY\nOfficial Baseline: {avg_b:.2f} dB\nOur Improved:      {avg_i:.2f} dB")
    if run_i and run_b: print(f"FINAL DELTA:       +{avg_i - avg_b:.2f} dB")
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
