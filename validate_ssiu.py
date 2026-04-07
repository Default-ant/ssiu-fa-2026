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
        self.n_feats = 16
        self.n_blocks = 10
        self.colors = 3

def calculate_psnr(img1, img2, border=4):
    """
    Standard Academic PSNR (Y-channel, border shaved).
    Uses BT.601 coefficients (MATLAB Standard).
    """
    if img1.shape != img2.shape: return 0
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 1. Convert to Y-Channel (Academic Standard)
    # y = 16 + 0.2567*R + 0.5041*G + 0.0979*B (for 0-255 scale)
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    
    # 2. Shave borders (Standard for SR)
    if border > 0:
        y1 = y1[border:-border, border:-border]
        y2 = y2[border:-border, border:-border]
    
    # 3. Calculate MSE and PSNR (Peak 255)
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def validate(model_path, data_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Detect Dataset and SOTA Target
    ds_name = "set5" 
    if data_path:
        path_lower = data_path.lower()
        for key in SOTA_BENCHMARKS:
            if key in path_lower:
                ds_name = key
                break
    sota_target = SOTA_BENCHMARKS.get(ds_name, 32.64)
    print(f"🚀 Running Academic Live Benchmark (Dataset: {ds_name.upper()} | Device: {device})")
    
    # 2. Setup Models
    # Improved Model (RGB Based)
    model_i = ImprovedSSIUNet(upscale=4).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Improved weights {model_path} not found.")
        return
    model_i.load_state_dict(torch.load(model_path, map_location=device))
    model_i.eval()
    
    # Baseline Model (BGR Based as per official GitHub source)
    model_b = None
    baseline_weights = "pretrain_model/model_x4_290.pt"
    if BASELINE_AVAILABLE and os.path.exists(baseline_weights):
        print(f"✅ Found Local Baseline Weights. Running LIVE comparison (BGR Mode).")
        try:
            model_b = SSUFSRNet(BaselineArgs(scale=4)).to(device)
            model_b.load_state_dict(torch.load(baseline_weights, map_location=device))
            model_b.eval()
        except Exception as e:
            print(f"⚠️ Baseline init error: {e}")
            model_b = None
    else:
        print(f"⚠️ Using Paper Reference ({sota_target} dB) for baseline.")

    # 3. Setup Dataset
    test_dir = data_path if data_path else 'MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/dependencies/KAIR/testsets/set5'
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found!")
        return
        
    hr_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    if not hr_files:
        hr_subfolder = os.path.join(test_dir, 'HR')
        if os.path.exists(hr_subfolder):
            test_dir = hr_subfolder
            hr_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    
    if not hr_files:
        print(f"Error: No images found in {test_dir}")
        return
    
    psnrs_i = []
    psnrs_b = []
    print(f"\n--- {ds_name.upper()} ---")
    print(f"{'Image':<15} | {'Official SSIU':<15} | {'Our Improved':<15} | {'Delta':<8}")
    print("-" * 65)
    
    for hr_file in hr_files:
        hr_path = os.path.join(test_dir, hr_file)
        hr_bgr = cv2.imread(hr_path)
        if hr_bgr is None: continue
        
        # Crop for x4
        h, w, _ = hr_bgr.shape
        hr_bgr = hr_bgr[:h-(h%4), :w-(w%4), :]
        lr_bgr = cv2.resize(hr_bgr, (hr_bgr.shape[1]//4, hr_bgr.shape[0]//4), interpolation=cv2.INTER_CUBIC)
        
        # Standard input for models (0-1)
        lr_bgr_t = torch.from_numpy(lr_bgr.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        # Prepare RGB version for Improved model
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        lr_rgb_t = torch.from_numpy(lr_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        with torch.no_grad():
            # Improved Inference (RGB)
            sr_i_t = model_i(lr_rgb_t.to(device))
            sr_i_rgb = (sr_i_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            psnr_i = calculate_psnr(sr_i_rgb, hr_rgb) # HR is RGB
            psnrs_i.append(psnr_i)

            # Baseline Inference (BGR)
            if model_b:
                sr_b_t = model_b(lr_bgr_t.to(device))
                sr_b_bgr = (sr_b_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                # Baseline Y-channel from BGR output
                # Need to convert BGR back to BGR-ordered float for our Y function (which expects RGB)
                # Or just convert to RGB first for consistency
                sr_b_rgb = cv2.cvtColor(sr_b_bgr, cv2.COLOR_BGR2RGB)
                psnr_b = calculate_psnr(sr_b_rgb, hr_rgb)
                psnrs_b.append(psnr_b)

        b_val = psnr_b if model_b else sota_target
        b_str = f"{psnr_b:>10.2f} dB" if model_b else "   (Paper Ref) "
        print(f"{hr_file[:15]:<15} | {b_str} | {psnr_i:>10.2f} dB | +{psnr_i - b_val:>.2f}")

    avg_i = np.mean(psnrs_i)
    avg_b = np.mean(psnrs_b) if model_b else sota_target
    
    print("\n" + "="*45)
    print(f"📊 {ds_name.upper()} ACADEMIC SUMMARY")
    print(f"Official Baseline: {avg_b:.2f} dB")
    print(f"Our Improved:      {avg_i:.2f} dB")
    print(f"FINAL DELTA:       +{avg_i - avg_b:.2f} dB")
    
    if avg_i >= avg_b:
        print(f"STATUS: 🏆 BEAT SOTA!")
    print("="*45 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="ssiu-div2k.pth")
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    validate(args.model_path, args.data_path)
