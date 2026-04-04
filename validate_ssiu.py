import torch
import cv2
import numpy as np
import os
from ssiu_improved import ImprovedSSIUNet

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def validate(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedSSIUNet(upscale=4).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Weights {model_path} not found. Training might still be running.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_dir = 'MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/dependencies/KAIR/testsets/set5'
    hr_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
    
    psnrs = []
    print(f"\n--- VALIDATING FINAL SSIU-FA (2026) VS 2025 NTIRE SOTA ---")
    
    for hr_path in hr_paths:
        hr = cv2.imread(hr_path)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        
        # Ensure dimensions are multiples of 4 for Cubic Resizing
        h, w, _ = hr.shape
        hr = hr[:h-(h%4), :w-(w%4), :]
        
        lr = cv2.resize(hr, (hr.shape[1]//4, hr.shape[0]//4), interpolation=cv2.INTER_CUBIC)
        
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        lr_t = lr_t.to(device)
        
        with torch.no_grad():
            sr_t = model(lr_t)
            sr = sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            sr = (sr * 255.0).clip(0, 255).astype(np.uint8)
            
        psnr = calculate_psnr(sr, hr)
        psnrs.append(psnr)
        print(f"Image: {os.path.basename(hr_path)} | PSNR: {psnr:.2f} dB")

    avg_psnr = np.mean(psnrs)
    print(f"\n--- FINAL RESULTS ---")
    print(f"Average PSNR (Set5 x4): {avg_psnr:.2f} dB")
    print(f"SSIU (2025) SOTA Benchmark: 32.64 dB")
    status = "BEAT SOTA 🏆" if avg_psnr >= 32.64 else "PROGRESSING"
    print(f"Status: {status}")

if __name__ == "__main__":
    validate("final_improved_ssiu_project.pth")
