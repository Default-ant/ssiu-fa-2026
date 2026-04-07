import torch
import cv2
import numpy as np
import os
import sys

# Import both architectures
from ssiu_improved import ImprovedSSIUNet

# Add SSIU models to path to import the official architecture if present
# We check both the root and the SSIU folder
possible_ssiu_paths = [
    os.path.join(os.getcwd(), 'SSIU'),
    os.path.join(os.getcwd(), 'archive_v3_restoration')
]
for p in possible_ssiu_paths:
    if os.path.exists(p):
        sys.path.append(p)
        break

try:
    from models.SSUFSR_network import SSUFSRNet
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

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def validate(model_path, data_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Detect Dataset and SOTA Target
    ds_name = "set5" # Default
    if data_path:
        path_lower = data_path.lower()
        for key in SOTA_BENCHMARKS:
            if key in path_lower:
                ds_name = key
                break
    sota_target = SOTA_BENCHMARKS.get(ds_name, 32.64)
    print(f"🚀 Running Live Benchmark (Dataset: {ds_name.upper()} | Device: {device})")
    print(f"📂 Searching in: {os.getcwd()}")
    
    # 2. Live Discovery of Baseline
    model_b = None
    baseline_available = False
    baseline_weights = None
    
    # Check for empty submodule
    if os.path.exists('SSIU') and not os.listdir('SSIU'):
        print("⚠️ Warning: 'SSIU' folder exists but is EMPTY. (Missing submodule update?)")
    
    # Recursively find the network file
    import glob
    net_matches = glob.glob("**/SSUFSR_network.py", recursive=True)
    if net_matches:
        # Get the directory that contains the 'models' folder
        net_path = os.path.abspath(net_matches[0])
        net_dir = os.path.dirname(os.path.dirname(net_path))
        if net_dir not in sys.path:
            sys.path.insert(0, net_dir)
        try:
            from models.SSUFSR_network import SSUFSRNet
            baseline_available = True
        except ImportError as e:
            print(f"⚠️ Baseline dependency missing: {e}. Try: !pip install einops")
        except Exception as e:
            print(f"⚠️ Baseline arch error: {e}")

    # Recursively find weights
    weight_matches = glob.glob("**/model_x4_290.pt", recursive=True)
    if weight_matches:
        baseline_weights = weight_matches[0]

    # 3. Initialize Models
    # Improved Model
    model_i = ImprovedSSIUNet(upscale=4).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Improved weights {model_path} not found.")
        return
    model_i.load_state_dict(torch.load(model_path, map_location=device))
    model_i.eval()
    
    # Baseline Model (Live)
    if baseline_available and baseline_weights:
        print(f"✅ Found Baseline: {baseline_weights}")
        try:
            model_b = SSUFSRNet(BaselineArgs(scale=4)).to(device)
            model_b.load_state_dict(torch.load(baseline_weights, map_location=device))
            model_b.eval()
        except Exception as e:
            print(f"⚠️ Baseline load error: {e}")
            model_b = None
    else:
        print(f"⚠️ Using Paper Reference ({sota_target} dB) for baseline.")
        # Debug why it's skipping
        if not baseline_available: print("   (Arch not found)")
        if not baseline_weights: print("   (Weights not found)")

    # 3. Setup Dataset
    test_dir = data_path if data_path else 'MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/dependencies/KAIR/testsets/set5'
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found!")
        return

    hr_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    
    # Check for 'HR' subfolder
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
        hr = cv2.imread(hr_path)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        
        # Crop for x4
        h, w, _ = hr.shape
        hr = hr[:h-(h%4), :w-(w%4), :]
        lr = cv2.resize(hr, (hr.shape[1]//4, hr.shape[0]//4), interpolation=cv2.INTER_CUBIC)
        
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        lr_t = lr_t.to(device)
        
        with torch.no_grad():
            # Improved Inference
            sr_i_t = model_i(lr_t)
            sr_i = sr_i_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            sr_i = (sr_i * 255.0).clip(0, 255).astype(np.uint8)
            psnr_i = calculate_psnr(sr_i, hr)
            psnrs_i.append(psnr_i)

            # Baseline Inference (Live)
            if model_b:
                sr_b_t = model_b(lr_t)
                sr_b = sr_b_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                sr_b = (sr_b * 255.0).clip(0, 255).astype(np.uint8)
                psnr_b = calculate_psnr(sr_b, hr)
                psnrs_b.append(psnr_b)
            else:
                psnr_b = 0 # Placeholder

        b_val = psnr_b if model_b else sota_target
        delta = psnr_i - b_val
        b_str = f"{psnr_b:>10.2f} dB" if model_b else "   (Paper Ref) "
        print(f"{hr_file[:15]:<15} | {b_str} | {psnr_i:>10.2f} dB | +{delta:>.2f}")

    avg_i = np.mean(psnrs_i)
    avg_b = np.mean(psnrs_b) if model_b else sota_target
    
    print("\n" + "="*45)
    print(f"📊 {ds_name.upper()} FINAL SUMMARY")
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
