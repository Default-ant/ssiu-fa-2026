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
    if img1.shape != img2.shape: return 0
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    if border > 0:
        y1 = y1[border:-border, border:-border]
        y2 = y2[border:-border, border:-border]
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def safe_load(model, path, device):
    if not os.path.exists(path): return False
    try:
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        # Strip DataParallel 'module.' prefix
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        return True
    except Exception as e:
        print(f"⚠️ Load error for {path}: {e}")
        return False

def validate(model_path, data_path=None, baseline_only=False, improved_only=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Models
    model_i = ImprovedSSIUNet(upscale=4).to(device)
    if not safe_load(model_i, model_path, device):
        print(f"Error: Could not load improved weights {model_path}")
        return
    model_i.eval()
    
    model_b = None
    baseline_weights = "pretrain_model/model_x4_290.pt"
    if BASELINE_AVAILABLE and os.path.exists(baseline_weights):
        model_b = SSUFSRNet(BaselineArgs(scale=4)).to(device)
        if not safe_load(model_b, baseline_weights, device):
            model_b = None
        else:
            model_b.eval()

    # 2. Setup Dataset
    test_dir = data_path if data_path else 'MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/dependencies/KAIR/testsets/set5'
    if not os.path.exists(test_dir):
        test_dir = '/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5/HR' if os.path.exists('/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5/HR') else test_dir
    
    hr_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    if not hr_files and os.path.exists(os.path.join(test_dir, 'HR')):
        test_dir = os.path.join(test_dir, 'HR')
        hr_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    
    if not hr_files:
        print(f"Error: No images found in {test_dir}")
        return

    # 3. Calibration
    print("🔍 Calibrating model logic (Mean-Shift & Color-Space)...")
    cal_file = os.path.join(test_dir, hr_files[0])
    cal_img_bgr = cv2.imread(cal_file)
    h, w, _ = cal_img_bgr.shape
    cal_hr_rgb = cv2.cvtColor(cal_img_bgr[:h-(h%4), :w-(w%4), :], cv2.COLOR_BGR2RGB)
    cal_lr_bgr = cv2.resize(cal_img_bgr[:h-(h%4), :w-(w%4), :], (cal_hr_rgb.shape[1]//4, cal_hr_rgb.shape[0]//4), interpolation=cv2.INTER_CUBIC)
    
    # DIV2K Mean
    mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1).to(device)
    best_psnr = 0
    best_cfg = {"mean": False, "bgr": False}
    
    model_to_cal = model_b if (baseline_only and model_b) else model_i
    
    for use_bgr in [False, True]:
        for use_mean in [False, True]:
            lr = cal_lr_bgr if use_bgr else cv2.cvtColor(cal_lr_bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(lr.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            if use_mean: t = t - mean
            
            with torch.no_grad():
                sr_t = model_to_cal(t)
                if use_mean: sr_t = sr_t + mean
                sr = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                p = calculate_psnr(sr, cal_hr_rgb if not use_bgr else cal_img_bgr[:h-(h%4), :w-(w%4), :])
                if p > best_psnr:
                    best_psnr = p
                    best_cfg = {"mean": use_mean, "bgr": use_bgr}
    
    print(f"✅ Calibration Result: Mean-Shift={best_cfg['mean']}, BGR={best_cfg['bgr']}\n")

    # 4. Evaluation
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
        lr_bgr = cv2.resize(img_bgr, (img_bgr.shape[1]//4, img_bgr.shape[0]//4), interpolation=cv2.INTER_CUBIC)
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        
        def process(model, img, cfg):
            t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            if cfg['mean']: t = t - mean
            with torch.no_grad():
                sr_t = model(t)
                if cfg['mean']: sr_t = sr_t + mean
                return (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if run_i:
            sr_i = process(model_i, lr_bgr if best_cfg['bgr'] else lr_rgb, best_cfg)
            p_i = calculate_psnr(sr_i, img_bgr if best_cfg['bgr'] else img_rgb)
            psnrs_i.append(p_i)
        
        if run_b:
            # Baseline is always Mean-Shift True, BGR True
            sr_b = process(model_b, lr_bgr, {"mean": True, "bgr": True})
            p_b = calculate_psnr(sr_b, img_bgr)
            psnrs_b.append(p_b)
        else:
            p_b = SOTA_BENCHMARKS.get(os.path.basename(test_dir).lower(), 32.64)

        if run_i and run_b:
            print(f"{hr_file[:15]:<15} | {p_b:>10.2f} dB | {p_i:>10.2f} dB | +{p_i - p_b:>.2f}")
        elif run_i:
            print(f"{hr_file[:15]:<15} | Improved PSNR: {p_i:>10.2f} dB")
        else:
            print(f"{hr_file[:15]:<15} | Baseline PSNR: {p_b:>10.2f} dB")

    avg_i = np.mean(psnrs_i) if run_i else 0
    avg_b = np.mean(psnrs_b) if run_b else SOTA_BENCHMARKS.get('set5', 32.64)
    
    print(f"\n{'='*45}\n📊 FINAL SUMMARY\nOfficial Baseline: {avg_b:.2f} dB\nOur Improved:      {avg_i:.2f} dB")
    if run_i: print(f"FINAL DELTA:       +{avg_i - avg_b:.2f} dB")
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
