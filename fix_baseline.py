import torch
import cv2
import numpy as np
import os
import torch.nn.functional as F

# Import official architecture
from ssiu_official import SSUFSRNet

class BaselineArgs:
    def __init__(self, n_feats=64, n_blocks=10):
        self.scale = 4
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        self.colors = 3

def calculate_psnr(img1, img2):
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    # Target Y-channel for academic precision
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    # Mandatory 4-pixel shave
    y1 = y1[4:-4, 4:-4]
    y2 = y2[4:-4, 4:-4]
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def brute_force():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = "pretrain_model/model_x4_290.pt"
    
    # Target Image (Set5 Baby)
    hr_path = "/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5/HR/baby.png"
    if not os.path.exists(hr_path):
        hr_path = "MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/dependencies/KAIR/testsets/set5/HR/baby.png"
    
    img_bgr = cv2.imread(hr_path)
    h, w, _ = img_bgr.shape
    img_bgr = img_bgr[:h-(h%4), :w-(w%4), :]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Try different LR sources
    lr_official_dir = hr_path.replace("HR", "LR").replace("baby.png", "babyx4.png")
    if os.path.exists(lr_official_dir):
        lr_img_bgr = cv2.imread(lr_official_dir)
        print("✅ Found official LR image.")
    else:
        lr_img_bgr = cv2.resize(img_bgr, (w//4, h//4), interpolation=cv2.INTER_LINEAR) # Trying Linear too
        print("⚠️ Generating LR image (Bilinear).")

    # Try Different Architectures
    for n_blocks in [10]:
        print(f"Testing Arch: 64 Feats, {n_blocks} Blocks...")
        model = SSUFSRNet(BaselineArgs(n_feats=64, n_blocks=n_blocks)).to(device)
        ckpt = torch.load(weights_path, map_location=device)
        sd = ckpt.get('model_state_dict', ckpt)
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        
        try:
            model.load_state_dict(new_sd, strict=True)
            print("✅ Weights loaded perfectly (Strict=True)")
        except Exception as e:
            print(f"⚠️ Strict load failed: {e}")
            model.load_state_dict(new_sd, strict=False)

        model.eval()
        
        mean_val = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1).to(device)

        with torch.no_grad():
            for use_bgr in [False, True]:
                for use_mean in [False, True]:
                    for in_scale in [1.0, 255.0]:
                        inp = lr_img_bgr if use_bgr else cv2.cvtColor(lr_img_bgr, cv2.COLOR_BGR2RGB)
                        target = img_bgr if use_bgr else img_rgb
                        
                        t = torch.from_numpy(inp.copy()).permute(2, 0, 1).float().unsqueeze(0) * (in_scale / 255.0)
                        if use_mean: t = t - (mean_val * in_scale)
                        
                        sr_t = model(t.to(device))
                        if use_mean: sr_t = sr_t + (mean_val * in_scale)
                        
                        sr = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * (255.0 / in_scale)).clip(0, 255).astype(np.uint8)
                        
                        p = calculate_psnr(sr, target)
                        print(f"BGR:{use_bgr} | Mean:{use_mean} | Scal:{in_scale} | PSNR:{p:.2f}")
                        
                        if p > 32.5:
                            print(f"\n🚀 SUCCESS! Winner row found. PSNR: {p:.2f} dB")
                            return

if __name__ == "__main__":
    brute_force()
