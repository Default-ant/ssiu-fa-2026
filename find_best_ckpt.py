"""
Find Best Checkpoint
====================
Automatically evaluates all saved checkpoints in a directory to find 
which iteration achieved the highest PSNR on the validation set.

Useful for catching the exact moment before the model overfits, or if
the 50,000th iteration is lower than the 40,000th due to learning rate drops.
"""
import os
import glob
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='.', help='Directory with .pth checkpoints')
    parser.add_argument('--data_path', type=str, required=True, help='Path to validation dataset (Set5 HR)')
    args = parser.parse_args()

    # Find all iter checkpoints
    checkpoints = glob.glob(os.path.join(args.ckpt_dir, '*iter_*.pth'))
    # Also grab final if exists
    final = glob.glob(os.path.join(args.ckpt_dir, '*final*.pth'))
    checkpoints.extend(final)
    
    if not checkpoints:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return

    # Sort numerically by iteration if possible
    def get_iter(path):
        name = os.path.basename(path)
        try:
            return int(''.join(filter(str.isdigit, name)))
        except:
            return 999999
            
    checkpoints.sort(key=get_iter)

    print("=" * 60)
    print(" SWEEPING CHECKPOINTS FOR OPTIMAL PSNR ")
    print("=" * 60)

    best_psnr = 0.0
    best_ckpt = None

    for ckpt in checkpoints:
        name = os.path.basename(ckpt)
        cmd = ['python', 'eval.py', '--model_path', ckpt, '--data_path', args.data_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        psnr = None
        for line in result.stdout.split('\n'):
            if "AVERAGE" in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    try:
                        psnr = float(parts[1].replace('dB', '').strip())
                    except ValueError:
                        pass
                break
                
        if psnr is not None:
            print(f"[{name:<25s}] -> PSNR: {psnr:.2f} dB")
            if psnr > best_psnr:
                best_psnr = psnr
                best_ckpt = name
        else:
            print(f"[{name:<25s}] -> Evaluation failed or PSNR not found.")

    print("=" * 60)
    if best_ckpt:
        print(f"🏆 BEST CHECKPOINT: {best_ckpt} with {best_psnr:.2f} dB")
    print("=" * 60)

if __name__ == '__main__':
    main()
