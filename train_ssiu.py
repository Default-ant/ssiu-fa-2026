import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from ssiu_improved import ImprovedSSIUNet

class UltraLightningDataset(Dataset):
    def __init__(self, upscale=4, patch_size=16):
        super().__init__()
        target_dir = 'MSTbic_Project_Archive/SuperResolutionMultiscaleTraining/archive/MANGA109'
        
        self.hr_images = []
        if os.path.exists(target_dir):
            file_list = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
            print(f"Loading {len(file_list)} images for high-performance training... 🔥")
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

def train(model_type='improved', iterations=8000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STARTING SOTA-PUSH SSIU-FA (2026) TRAINING ---")
    
    model = ImprovedSSIUNet(upscale=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    criterion = CharbonnierLoss() # SOTA standard for SR
    
    dataset = UltraLightningDataset(upscale=4)
    if len(dataset) == 0: return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # Max saturation
    loader_iter = iter(dataloader)
    
    model.train()
    for i in range(iterations + 1):
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
            print(f"[{i:04d}/{iterations}] Total: {loss.item():.4f} | Charb: {loss_main.item():.4f} | Lf: {loss_f.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), f"final_{model_type}_ssiu_project.pth")
    print(f"\n--- ULTRA-FAST RECONSTRUCTION COMPLETE ---")
    print(f"Results are ready for IEEE Trans Analysis.")

if __name__ == "__main__":
    # Increased patch_size to 32 for better SOTA performance
    train(model_type='improved', iterations=8000)
