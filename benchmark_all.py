"""
Benchmark All Datasets
======================
Runs the evaluation pipeline across all 5 standard SR benchmarks:
Set5, Set14, BSD100, Urban100, and Manga109.

Outputs a consolidated JSON summary and prints a formatted markdown table
ready for your paper.
"""
import os
import argparse
import subprocess
import glob
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to your .pth file')
    parser.add_argument('--baseline_path', type=str, default=None, help='Path to baseline .pt')
    parser.add_argument('--datasets_dir', type=str, required=True, help='Root directory containing Set5, Set14, etc.')
    args = parser.parse_args()

    datasets = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']
    results = {}

    print("=" * 60)
    print(f" FULL BENCHMARK SUITE - MODEL: {os.path.basename(args.model_path)}")
    print("=" * 60)

    for ds in datasets:
        # Try to find the dataset path robustly (could be Set5, set5, Set5/HR)
        ds_path = None
        for root, dirs, _ in os.walk(args.datasets_dir):
            if os.path.basename(root).lower() == ds.lower():
                ds_path = os.path.join(root, 'HR') if 'HR' in dirs else root
                break
        
        if not ds_path:
            # Fallback path logic
            potential_path = os.path.join(args.datasets_dir, ds)
            if os.path.exists(potential_path):
                ds_path = potential_path

        if not ds_path or not os.path.exists(ds_path):
            print(f"[SKIP] {ds:<10s}: Could not find directory.")
            continue

        print(f"Evaluating {ds}...")
        
        # Build command calling our trusted eval.py
        cmd = ['python', 'eval.py', '--model_path', args.model_path, '--data_path', ds_path]
        if args.baseline_path:
            cmd.extend(['--baseline_path', args.baseline_path])

        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse stdout for the AVERAGE line
        psnr, ssim = None, None
        base_psnr = None
        
        for line in result.stdout.split('\n'):
            if "AVERAGE" in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    try:
                        # parts[1] looks like "32.64 dB"
                        psnr = float(parts[1].replace('dB', '').strip())
                        ssim = float(parts[2].strip())
                        
                        if len(parts) >= 4 and parts[3] != 'N/A':
                            base_psnr = float(parts[3].replace('dB', '').strip())
                    except ValueError:
                        pass
                break
        
        results[ds] = {
            'Ours_PSNR': psnr,
            'Ours_SSIM': ssim,
            'Base_PSNR': base_psnr
        }

    print("\n" + "=" * 60)
    print(" 🏆 FINAL PAPER RESULTS TABLE ")
    print("=" * 60)
    
    # Format exactly like a paper table
    df = pd.DataFrame.from_dict(results, orient='index')
    print(df.to_markdown())
    print("=" * 60)

if __name__ == '__main__':
    main()
