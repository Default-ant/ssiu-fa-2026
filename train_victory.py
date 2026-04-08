import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import time
from ssiu_v3 import SSIUV3Net

class VictoryDataset(Dataset):
    def __init__(self, data_path, upscale=4, patch_size=48):
        super().__init__()
        self.hr_images = []
        if not os.path.exists(data_path):
            raise Exception(f"Data path {data_path} not found!")
            
        # Common Kaggle DIV2K paths often have 'DIV2K_train_HR'
        search_path = data_path
        if os.path.exists(os.path.join(data_path, 'DIV2K_train_HR')):
            search_path = os.path.join(data_path, 'DIV2K_train_HR')
            
        file_list = sorted([os.path.join(search_path, f) for f in os.listdir(search_path) if f.lower().endswith(('.png', '.jpg', '.bmp'))])[:800]
        print(f"🔥 Loading {len(file_list)} DIV2K images for Victory Run training...")
        
        # Load images into memory for speed (Kaggle has 16-30GB RAM)
        for i, f in enumerate(file_list):
            img = cv2.imread(f)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.hr_images.append(img)
            if i % 100 == 0: print(f"Loaded {i} images...")
            
        self.upscale = upscale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.hr_images) * 100 # Artificial length for better epoch logic

    def __getitem__(self, idx):
        hr = self.hr_images[idx % len(self.hr_images)]
        h, w, _ = hr.shape
        p_hr = self.patch_size * self.upscale
        
        # Random Crop
        x = np.random.randint(0, w - p_hr)
        y = np.random.randint(0, h - p_hr)
        hr_crop = hr[y : y + p_hr, x : x + p_hr]
        
        # Augmentation
        aug = np.random.randint(0, 4)
        if aug == 1: hr_crop = np.flipud(hr_crop)
        elif aug == 2: hr_crop = np.fliplr(hr_crop)
        elif aug == 3: hr_crop = np.rot90(hr_crop)
        
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
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

def train(data_path, iterations=50000, batch_size=16, resume=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting SSIU-V3 Victory Training on {device}...")
    
    model = SSIUV3Net(upscale=4).to(device)
    if resume and os.path.exists(resume):
        print(f"📂 Resuming from {resume}...")
        model.load_state_dict(torch.load(resume, map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    criterion = CharbonnierLoss()
    
    dataset = VictoryDataset(data_path=data_path, upscale=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_iter = iter(dataloader)
    
    start_time = time.time()
    for i in range(1, iterations + 1):
        try:
            lr, hr = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            lr, hr = next(loader_iter)
            
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        scheduler.step()
        
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Iter: {i:05d}/{iterations} | Loss: {loss.item():.6f} | Time: {elapsed:.1f}s")
            start_time = time.time()
            
        if i % 5000 == 0:
            path = f"ssiu_v3_iter_{i}.pth"
            torch.save(model.state_dict(), path)
            print(f"💾 Checkpoint saved: {path}")

    torch.save(model.state_dict(), "ssiu_v3_final_victory.pth")
    print("✅ Training Complete! Model saved as ssiu_v3_final_victory.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--iters', type=int, default=50000)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    train(args.data_path, iterations=args.iters, resume=args.resume)
