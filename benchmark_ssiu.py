import torch
import time
import numpy as np
from ssiu_improved import ImprovedSSIUNet

def benchmark_efficiency(upscale=4, iterations=100, warm_up=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- SSIU-V2 EFFICIENCY BENCHMARK ({device}) ---")
    
    # 1. Initialize Model
    model = ImprovedSSIUNet(upscale=upscale).to(device)
    model.eval()
    
    # 2. Get Parameter Count
    params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"Total Parameters: {params:.2f}K (Target SSIU Paper: 794K)")

    # 3. Create Dummy Input (480x320 as per Table II footnote)
    # LR input (size // 4)
    lr_input = torch.randn(1, 3, 320 // upscale, 480 // upscale).to(device)
    
    # 4. WARM UP
    print(f"Warming up for {warm_up} iterations...")
    for _ in range(warm_up):
        with torch.no_grad():
            _ = model(lr_input)
    
    # 5. MEASURE GPU MEMORY
    peak_mem = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(lr_input)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
    
    # 6. MEASURE INFERENCE TIME
    print(f"Measuring average time over {iterations} iterations...")
    timings = []
    
    if torch.cuda.is_available():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(iterations):
                starter.record()
                _ = model(lr_input)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
    else:
        # Fallback for CPU
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(lr_input)
            end = time.perf_counter()
            timings.append((end - start) * 1000) # ms
            
    avg_time = np.mean(timings)
    std_time = np.std(timings)

    # 7. FINAL REPORT
    print("\n" + "="*30)
    print("📊 BENCHMARK RESULTS (TABLE II)")
    print(f"Model: SSIU-V2 (SSIU-FA)")
    print(f"Device: {device}")
    print(f"Input: 480x320 -> Output: 1920x1280 (x4)")
    print(f"Average Time: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    if torch.cuda.is_available():
        print(f"Peak GPU Mem: {peak_mem:.2f} MB")
    else:
        print(f"Memory: N/A (Run on GPU for memory stats)")
    print("="*30)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA not available. Results will not match research standards.")
    benchmark_efficiency()
