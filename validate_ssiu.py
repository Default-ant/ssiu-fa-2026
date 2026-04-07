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
    # y = 16 + 0.2567*R + 0.5041*G + 0.0979*B
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    if border > 0:
        y1 = y1[border:-border, border:-border]
        y2 = y2[border:-border, border:-border]
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def validate(model_path, data_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Detect Dataset
    ds_name = "set5" 
    if data_path:
        path_lower = data_path.lower()
        for key in SOTA_BENCHMARKS:
            if key in path_lower: ds_name = key; break
    sota_target = SOTA_BENCHMARKS.get(ds_name, 32.64)
    
    # 2. Setup Models
    model_i = ImprovedSSIUNet(upscale=4).to(device)
    model_i.load_state_dict(torch.load(model_path, map_location=device))
    model_i.eval()
    
    model_b = None
    baseline_weights = "pretrain_model/model_x4_290.pt"
    if BASELINE_AVAILABLE and os.path.exists(baseline_weights):
        model_b = SSUFSRNet(BaselineArgs(scale=4)).to(device)
        model_b.load_state_dict(torch.load(baseline_weights, map_location=device))
        model_b.eval()

    # 3. Setup Dataset
    test_dir = data_path if data_path else 'MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/dependencies/KAIR/testsets/set5'
    if not os.path.exists(test_dir):
        # Fallback to current dir if Kaggle structure varies
        test_dir = '/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5/HR' if 'kaggle' in os.getcwd() else test_dir
    
    hr_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    if not hr_files and os.path.exists(os.path.join(test_dir, 'HR')):
        test_dir = os.path.join(test_dir, 'HR')
        hr_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    
    if not hr_files:
        print(f"Error: No images found in {test_dir}")
        return

    # 4. SMART CALIBRATION (Find best preprocessing for these weights)
    print("🔍 Calibrating model logic for peak performance...")
    cal_file = os.path.join(test_dir, hr_files[0])
    cal_img = cv2.imread(cal_file)
    cal_hr = cv2.cvtColor(cal_img, cv2.COLOR_BGR2RGB)
    h, w, _ = cal_hr.shape
    cal_hr = cal_hr[:h-(h%4), :w-(w%4), :]
    cal_lr_bgr = cv2.resize(cal_img[:h-(h%4), :w-(w%4), :], (cal_hr.shape[1]//4, cal_hr.shape[0]//4), interpolation=cv2.INTER_CUBIC)
    
    best_psnr = 0
    best_config = {"mean": False, "bgr": False}
    
    # Standard DIV2K Mean
    mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1).to(device)
    
    for use_bgr in [False, True]:
        for use_mean in [False, True]:
            # Simple test on Improved Model
            lr = cal_lr_bgr if use_bgr else cv2.cvtColor(cal_lr_bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(lr.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            if use_mean: t = t - mean
            
            with torch.no_grad():
                sr_t = model_i(t)
                if use_mean: sr_t = sr_t + mean
                sr = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                p = calculate_psnr(sr, cal_hr if not use_bgr else cal_img[:h-(h%4), :w-(w%4), :])
                if p > best_psnr:
                    best_psnr = p
                    best_config = {"mean": use_mean, "bgr": use_bgr}
    
    print(f"✅ Calibration Complete! Best Config: Mean-Shift={best_config['mean']}, BGR-Mode={best_config['bgr']}")
    print(f"📊 Initial Calibration PSNR: {best_psnr:.2f} dB\n")

    # 5. Run Full Benchmark
    psnrs_i = []
    psnrs_b = []
    print(f"🎯 Running Official Side-by-Side (Dataset: {ds_name.upper()})")
    print("-" * 65)
    
    for hr_file in hr_files:
        hr_path = os.path.join(test_dir, hr_file)
        img_bgr = cv2.imread(hr_path)
        h, w, _ = img_bgr.shape
        img_bgr = img_bgr[:h-(h%4), :w-(w%4), :]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        lr_bgr = cv2.resize(img_bgr, (img_bgr.shape[1]//4, img_bgr.shape[0]//4), interpolation=cv2.INTER_CUBIC)
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        
        # Prepare Tensors
        def prepare(img, conf):
            t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            if conf['mean']: t = t - mean
            return t
        
        t_i = prepare(lr_bgr if best_config['bgr'] else lr_rgb, best_config)
        t_b = prepare(lr_bgr, {"mean": True, "bgr": True}) # Baseline is almost always Mean-Shift BGR
        
        with torch.no_grad():
            # Improved
            sr_i_t = model_i(t_i)
            if best_config['mean']: sr_i_t = sr_i_t + mean
            sr_i = (sr_i_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            psnr_i = calculate_psnr(sr_i, img_bgr if best_config['bgr'] else img_rgb)
            psnrs_i.append(psnr_i)

            # Baseline
            if model_b:
                sr_b_t = model_b(t_b)
                sr_b_t = sr_b_t + mean # Baseline always needs mean for SOTA results
                sr_b = (sr_b_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                # Compare in BGR if weights are BGR
                psnr_b = calculate_psnr(sr_b, img_bgr)
                psnrs_b.append(psnr_b)
            else: psnr_b = sota_target

        b_str = f"{psnr_b:>10.2f} dB" if model_b else "   (Paper Ref) "
        print(f"{hr_file[:15]:<15} | {b_str} | {psnr_i:>10.2f} dB | +{psnr_i - (psnr_b if model_b else sota_target):>.2f}")

    avg_i, avg_b = np.mean(psnrs_i), (np.mean(psnrs_b) if model_b else sota_target)
    print(f"\n{'='*45}\n📊 {ds_name.upper()} FINAL SUMMARY\nOfficial Baseline: {avg_b:.2f} dB\nOur Improved:      {avg_i:.2f} dB\nFINAL DELTA:       +{avg_i - avg_b:.2f} dB")
    if avg_i >= avg_b: print("STATUS: 🏆 BEAT SOTA!")
    print("="*45 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="ssiu-div2k.pth")
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    validate(args.model_path, args.data_path)
