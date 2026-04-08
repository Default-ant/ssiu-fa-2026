import torch
import cv2
import numpy as np
import os
import sys
from PIL import Image

try:
    from ssiu_official import SSUFSRNet
except:
    sys.exit(1)

def calculate_psnr(img1, img2, border=4):
    if img1.shape != img2.shape: return 0
    y1 = 16.0 + (65.481 * img1[..., 0] + 128.553 * img1[..., 1] + 24.966 * img1[..., 2]) / 255.0
    y2 = 16.0 + (65.481 * img2[..., 0] + 128.553 * img2[..., 1] + 24.966 * img2[..., 2]) / 255.0
    y1, y2 = y1[border:-border, border:-border], y2[border:-border, border:-border]
    mse = np.mean((y1 - y2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from collections import namedtuple
    Args = namedtuple('Args', ['scale', 'n_feats', 'n_blocks', 'colors'])
    model = SSUFSRNet(Args(4, 64, 10, 3)).to(device)
    ckpt = torch.load("pretrain_model/model_x4_290.pt", map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    model.eval()

    # Load one image for sweeping (baby.png)
    data_path = "/kaggle/input/datasets/chenqizhou/set5-hr-lr/Set5"
    img_path = None
    for root, _, files in os.walk(data_path):
        for f in files:
            if "baby" in f.lower() and f.endswith(('.png', '.jpg')):
                img_path = os.path.join(root, f); break
        if img_path: break
    
    if not img_path:
        print("Error: baby.png not found for sweep.")
        return

    img_bgr = cv2.imread(img_path)
    h, w, _ = img_bgr.shape
    img_bgr = img_bgr[:h-(h%4), :w-(w%4), :]
    hr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    lr_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).resize((w//4, h//4), resample=Image.BICUBIC)
    lr_rgb = np.array(lr_pil)
    lr_bgr = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2BGR)
    
    mean_rgb = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1).to(device)
    mean_bgr = torch.Tensor([0.4040, 0.4371, 0.4488]).view(1, 3, 1, 1).to(device)

    print(f"🕵️‍♂️ SWEEPING CONFIGURATIONS FOR BABY.PNG (Target: ~33.91 dB)")
    print("-" * 60)
    
    configs = [
        ("RGB", "No Mean", 1.0),
        ("RGB", "Mean", 1.0),
        ("BGR", "No Mean", 1.0),
        ("BGR", "Mean", 1.0),
        ("RGB", "No Mean", 255.0),
        ("RGB", "Mean", 255.0),
        ("BGR", "No Mean", 255.0),
        ("BGR", "Mean", 255.0),
    ]

    for color, m_type, scale in configs:
        lr = lr_rgb if color == "RGB" else lr_bgr
        m = mean_rgb if color == "RGB" else mean_bgr
        
        t = torch.from_numpy(lr.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)
        t = t * (scale / 255.0)
        if m_type == "Mean": t = t - (m * scale)
        
        with torch.no_grad():
            sr_t = model(t)
            if m_type == "Mean": sr_t = sr_t + (m * scale)
            sr = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * (255.0 / scale)).clip(0, 255).astype(np.uint8)
            
            # Check PSNR in RGB
            test_sr = sr if color == "RGB" else cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
            psnr = calculate_psnr(test_sr, hr_rgb)
            print(f"{color:<4} | {m_type:<8} | Scale: {scale:<5} | PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    sweep()
