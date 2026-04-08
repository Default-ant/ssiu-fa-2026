import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import cv2
import numpy as np
import os
import sys
from ssiu_improved import ImprovedSSIUNet

class DIV2KSOTADataset(Dataset):
    def __init__(self, data_path=None, upscale=4, patch_size=64):
        super().__init__()
        self.target_dir = data_path if data_path else '/kaggle/input/datasets/harshraone/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR'
        self.file_list = []
        if os.path.exists(self.target_dir):
            self.file_list = sorted([os.path.join(self.target_dir, f) for f in os.listdir(self.target_dir) if f.endswith(('.png', '.jpg'))])
            print(f"Loading {len(self.file_list)} images for SOTA training... 🖼️")
        
        self.upscale = upscale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        hr = cv2.imread(self.file_list[idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        h, w, _ = hr.shape
        
        p_hr = self.patch_size * self.upscale
        x = np.random.randint(0, w - p_hr)
        y = np.random.randint(0, h - p_hr)
        hr_crop = hr[y : y + p_hr, x : x + p_hr]
        
        # Sota Augmentations
        aug = np.random.randint(0, 8)
        if aug == 1: hr_crop = np.flipud(hr_crop)
        elif aug == 2: hr_crop = np.fliplr(hr_crop)
        elif aug == 3: hr_crop = np.rot90(hr_crop)
        elif aug == 4: hr_crop = np.rot90(hr_crop, 2)
        elif aug == 5: hr_crop = np.rot90(hr_crop, 3)
        
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

def train(iterations=50000, data_path=None, resume_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STARTING FINAL SOTA RUN ({iterations} Iterations) ---")
    
    # Matching Verified Baseline architecture: 9 blocks, 64 features
    model = ImprovedSSIUNet(upscale=4).to(device)
    scaler = torch.amp.GradScaler('cuda')
    
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        model.load_state_dict(torch.load(resume_path, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    criterion = CharbonnierLoss()
    
    dataset = DIV2KSOTADataset(data_path=data_path, upscale=4, patch_size=64)
    if len(dataset) == 0: return
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    model.train()
    loader_iter = iter(dataloader)
    
    for i in range(1, iterations + 1):
        try:
            lr, hr = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            lr, hr = next(loader_iter)
            
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            sr = model(lr)
            loss = criterion(sr, hr)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if i % 100 == 0:
            print(f"🚀 Progress: {i}/{iterations} | Loss: {loss.item():.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if i % 5000 == 0 or i == iterations:
            path = f"ssiu_improved_sota_iter_{i}.pth"
            torch.save(model.state_dict(), path)
            print(f"💾 Checkpoint: {path}")

    print("--- SOTA TRAINING COMPLETE ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    train(iterations=args.iterations, data_path=args.data_path, resume_path=args.resume)
