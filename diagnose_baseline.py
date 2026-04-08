import torch
import cv2
import numpy as np
import os
import torch.nn.functional as F

from ssiu_official import SSUFSRNet

class BaselineArgs:
    def __init__(self, scale=4):
        self.scale = scale
        self.n_feats = 64
        self.n_blocks = 10
        self.colors = 3

def calculate_psnr(img1, img2):
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    # Target Y-channel for academic precision
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    y1, y2 = y1[4:-4, 4:-4], y2[4:-4, 4:-4]
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def run_diagnostics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = "pretrain_model/model_x4_290.pt"
    
    if not os.path.exists(weights_path):
        print("❌ Baseline weights not found!")
        return

    # Load Model
    model = SSUFSRNet(BaselineArgs()).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(new_sd)
    model.eval()

    # Load Test Image (Set5 Baby)
    hr_path = "/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5/HR/baby.png"
    if not os.path.exists(hr_path):
        # Fallback search
        hr_path = "MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/dependencies/KAIR/testsets/set5/HR/baby.png"

    img_bgr = cv2.imread(hr_path)
    h, w, _ = img_bgr.shape
    img_bgr = img_bgr[:h-(h%4), :w-(w%4), :]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    lr_bgr = cv2.resize(img_bgr, (w//4, h//4), interpolation=cv2.INTER_CUBIC)
    lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)

    print(f"🕵️‍♂️ Starting Brute-Force Diagnostic on {os.path.basename(hr_path)}...")
    print(f"{'BGR':<5} | {'Scale':<6} | {'Logic':<15} | {'PSNR'}")
    print("-" * 45)

    with torch.no_grad():
        for use_bgr in [False, True]:
            for in_scale in [1.0, 255.0]:
                for logic in ["Standard", "Remove-Skip"]:
                    # Prepare Input
                    inp = lr_bgr if use_bgr else lr_rgb
                    target = img_bgr if use_bgr else img_rgb
                    
                    t = torch.from_numpy(inp.copy()).permute(2, 0, 1).float().unsqueeze(0) * (in_scale / 255.0)
                    t = t.to(device)
                    
                    # Run Forward
                    out_t = model(t)
                    
                    # Logic: Standard (Model output) vs Remove-Skip (Model is adding skip, maybe don't want it?)
                    if logic == "Remove-Skip":
                        # Manually subtract the bilinear skip if it was added incorrectly
                        skip = F.interpolate(t, scale_factor=4, mode='bilinear', align_corners=False)
                        out_t = out_t - skip
                    
                    # Convert to SR image
                    sr = (out_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * (255.0 / in_scale)).clip(0, 255).astype(np.uint8)
                    
                    psnr = calculate_psnr(sr, target)
                    
                    bgr_str = "YES" if use_bgr else "NO"
                    print(f"{bgr_str:<5} | {in_scale:<6} | {logic:<15} | {psnr:.2f} dB")
                    
                    if psnr > 32:
                        print(f"🎉 FOUND IT! Config: BGR={use_bgr}, Scale={in_scale}, Logic={logic}")

if __name__ == "__main__":
    run_diagnostics()
