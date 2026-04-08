"""
Visualize Super-Resolution Results
==================================
Generates side-by-side visual comparisons of Bicubic vs Model vs HR.
Highlights detailed patches to prove visual quality for the paper.

Usage:
  python visualize_sr.py --model_path ssiu_fa_x4_final.pth --image_path datasets/Set5/HR/baby.png --output result_baby.png
"""
import torch
import cv2
import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ssiu_improved import ImprovedSSIUNet

def get_patch(img, x, y, size):
    """Safely extract a square patch."""
    h, w = img.shape[:2]
    # Bound checks
    x = max(0, min(x, w - size))
    y = max(0, min(y, h - size))
    return img[y:y+size, x:x+size]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output', type=str, default="sr_comparison.png")
    parser.add_argument('--patch_x', type=int, default=150, help='X coordinate for zoomed patch')
    parser.add_argument('--patch_y', type=int, default=150, help='Y coordinate for zoomed patch')
    parser.add_argument('--patch_size', type=int, default=60, help='Size of the zoomed patch')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    model = ImprovedSSIUNet(upscale=4).to(device)
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        print(f"Model path {args.model_path} not found.")
        return
    model.eval()

    # 2. Process Image
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        print(f"Could not load {args.image_path}")
        return
    
    h, w = img_bgr.shape[:2]
    h, w = h - (h % 4), w - (w % 4)
    img_bgr = img_bgr[:h, :w]
    hr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    lr_pil = Image.fromarray(hr_rgb).resize((w // 4, h // 4), resample=Image.BICUBIC)
    lr_rgb = np.array(lr_pil)
    bicubic_rgb = np.array(lr_pil.resize((w, h), resample=Image.BICUBIC))

    # 3. Model Inference
    lr_t = torch.from_numpy(lr_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        sr_t = model(lr_t)
    sr_rgb = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # 4. Generate Visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Show full HR image with bounding box
    ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax_main.imshow(hr_rgb)
    ax_main.set_title("Original HR Image", fontsize=14)
    ax_main.axis('off')
    
    # Add rectangle to show where we're zooming
    rect = Rectangle((args.patch_x, args.patch_y), args.patch_size, args.patch_size, 
                     linewidth=3, edgecolor='red', facecolor='none')
    ax_main.add_patch(rect)

    # Get Patches
    patch_bicubic = get_patch(bicubic_rgb, args.patch_x, args.patch_y, args.patch_size)
    patch_sr = get_patch(sr_rgb, args.patch_x, args.patch_y, args.patch_size)
    patch_hr = get_patch(hr_rgb, args.patch_x, args.patch_y, args.patch_size)

    # Plot Patches
    titles = ["Bicubic (x4)", "SSIU-FA (Ours)", "HR (Ground Truth)"]
    patches = [patch_bicubic, patch_sr, patch_hr]

    for i in range(3):
        ax = plt.subplot2grid((2, 3), (1, i))
        ax.imshow(patches[i])
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')
        # Add red border around the patches
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Visualization saved successfully to {args.output}")

if __name__ == "__main__":
    main()
