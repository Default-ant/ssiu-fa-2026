import torch
import torch.nn as nn
from ssiu_improved import ImprovedSSIUNet
import os

def migrate_24_to_28(src_path, dst_path):
    print(f"🚀 Migrating weights from 24-blocks to 28-blocks...")
    print(f"   Source: {src_path}")
    
    # 1. Load the 24-block checkpoint
    checkpoint = torch.load(src_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        src_state = checkpoint['model_state_dict']
    else:
        src_state = checkpoint
        
    # 2. Initialize the 28-block model
    # Note: Ensure the architecture defaults are updated in ssiu_improved.py or passed here
    model_28 = ImprovedSSIUNet(num_blocks=28)
    dst_state = model_28.state_dict()
    
    # 3. Map weights
    # Structure:
    # layers.0...layers.23 (24 blocks)
    # layers.0...layers.27 (28 blocks)
    
    matched_keys = 0
    for key in src_state.keys():
        if key in dst_state:
            # Check if dimensions match (they should for everything except maybe layer indices)
            if src_state[key].shape == dst_state[key].shape:
                dst_state[key] = src_state[key]
                matched_keys += 1
            else:
                print(f"⚠️  Dimension mismatch for {key}: {src_state[key].shape} vs {dst_state[key].shape}")
        else:
            print(f"❓ Key NOT in target: {key}")

    # 4. Success check
    print(f"✅ Matched {matched_keys} weight tensors.")
    print(f"   New blocks (layers.24 to layers.27) remain initialized with random weights.")
    
    # 5. Save the new starting weights
    torch.save(dst_state, dst_path)
    print(f"📂 Saved 28-block starting weights to: {dst_path}")
    
    params_k = sum(p.numel() for p in model_28.parameters()) / 1e3
    print(f"✨ Model Params: {params_k:.1f}K")

if __name__ == "__main__":
    SRC = "SSIU/ssiu_fa_24b_iter_40000.pth"
    DST = "ssiu_fa_28b_start.pth"
    if os.path.exists(SRC):
        migrate_24_to_28(SRC, DST)
    else:
        print(f"❌ Could not find source: {SRC}")
