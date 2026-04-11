import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from ssiu_improved import ImprovedSSIUNet

class UltraLightningDataset(Dataset):
    def __init__(self, data_path=None, upscale=4, patch_size=32):
        super().__init__()
        target_dir = data_path if data_path else 'MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/archive/MANGA109'
        
        self.hr_images = []
        if os.path.exists(target_dir):
            file_list = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
            print(f"Loading {len(file_list)} images for high-performance training from {target_dir}... 🔥")
            for f in file_list:
                img = cv2.imread(f)
                if img is not None:
                    # REMOVED resizing to 256 to keep full HR resolution
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.hr_images.append(img)
            print(f"Ready! Training on full-resolution HR sources. ⚡")
        else:
            print(f"Error: Path {target_dir} not found!")

        self.upscale = upscale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr = self.hr_images[idx]
        h, w, _ = hr.shape
        
        # Increased patch size for better structural learning
        p_hr = self.patch_size * self.upscale
        x = np.random.randint(0, w - p_hr)
        y = np.random.randint(0, h - p_hr)
        hr_crop = hr[y : y + p_hr, x : x + p_hr]
        
        # Data Augmentation: Flips and Rotations
        aug = np.random.randint(0, 8)
        if aug == 1: hr_crop = np.flipud(hr_crop)
        elif aug == 2: hr_crop = np.fliplr(hr_crop)
        elif aug == 3: hr_crop = np.rot90(hr_crop)
        elif aug == 4: hr_crop = np.rot90(hr_crop, 2)
        elif aug == 5: hr_crop = np.rot90(hr_crop, 3)
        elif aug == 6: hr_crop = np.flipud(np.rot90(hr_crop))
        
        lr_crop = cv2.resize(hr_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        
        hr_t = torch.from_numpy(hr_crop.copy()).permute(2, 0, 1).float() / 255.0
        lr_t = torch.from_numpy(lr_crop.copy()).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

def frequency_loss(sr, hr):
    sr_fft = torch.fft.rfft2(sr, norm='ortho')
    hr_fft = torch.fft.rfft2(hr, norm='ortho')
    return torch.mean(torch.abs(sr_fft - hr_fft))

def train(model_type='improved', iterations=25000, data_path=None, resume_path=None, upscale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STARTING FINAL SOTA-BEAT SSIU-V2 RUN ({upscale}x, {iterations} Iterations) ---")
    
    model = ImprovedSSIUNet(upscale=upscale).to(device)
    
    # RESUME FROM CHECKPOINT LOGIC
    start_iter = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}... 📂")
        model.load_state_dict(torch.load(resume_path, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    criterion = CharbonnierLoss() # SOTA standard for SR
    
    dataset = UltraLightningDataset(data_path=data_path, upscale=upscale)
    if len(dataset) == 0: return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    loader_iter = iter(dataloader)
    
    model.train()
    for i in range(start_iter, iterations + 1):
        try:
            lr, hr = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            lr, hr = next(loader_iter)
            
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        sr = model(lr)
        
        loss_main = criterion(sr, hr)
        loss_f = frequency_loss(sr, hr)
        loss = loss_main + 0.05 * loss_f
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % 100 == 0:
            pct = 100 * i / iterations
            print(f"🚀 Speed x{upscale}: {i:04d}/{iterations} ({pct:.0f}%) | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # PERIODIC CHECKPOINTS
        if i > 0 and i % 5000 == 0:
            ckpt_path = f"checkpoint_x{upscale}_{model_type}_iter_{i}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"\n💾 CHECKPOINT REACHED: Saved to {ckpt_path}\n")

    torch.save(model.state_dict(), f"ssiu_x{upscale}_final.pth")
    print(f"\n--- ULTRA-FAST RECONSTRUCTION x{upscale} COMPLETE ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=25000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--upscale', type=int, default=4)
    args = parser.parse_args()
    
    train(model_type='improved', iterations=args.iterations, data_path=args.data_path, resume_path=args.resume, upscale=args.upscale)
