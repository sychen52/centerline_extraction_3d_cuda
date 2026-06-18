import torch
import SimpleITK as sitk
import numpy as np
import time
import os
from centerline_extraction_3d_cuda import binary_thinning


def run_benchmark(input_path, num_runs=5):
    print(f"Loading {input_path}...")

    img_sitk = sitk.ReadImage(input_path)
    img_array = sitk.GetArrayFromImage(img_sitk)

    print(f"Original shape: {img_array.shape}, type: {img_array.dtype}")
    print(f"Original sum (volume): {np.sum(img_array > 0)}")

    # Keep a base CPU tensor for repeatable benchmarking
    base_tensor_cpu = torch.from_numpy(img_array > 0).to(torch.uint8)

    # 0. GPU Subgrid (8-color)
    print(f"\n--- 0. CUDA Thinning (GPU Subgrid 8-Color) - {num_runs} runs ---")
    times_subgrid = []

    for i in range(num_runs):
        tensor_subgrid = (
            base_tensor_cpu.clone()
        )  # Start with a CPU tensor to include copy overhead
        torch.cuda.synchronize()
        start_time = time.time()
        binary_thinning(tensor_subgrid, mode=0)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        times_subgrid.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.4f}s")

    cuda_time_subgrid = np.median(times_subgrid)

    # 1. GPU Hybrid (CPU Sync)
    print(f"\n--- 1. CUDA Thinning (Hybrid CPU-Sync) - {num_runs} runs ---")
    times_cpu_sync = []

    for i in range(num_runs):
        tensor_cpu_sync = base_tensor_cpu.clone()
        torch.cuda.synchronize()
        start_time = time.time()
        binary_thinning(tensor_cpu_sync, mode=1)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        times_cpu_sync.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.4f}s")

    cuda_time_cpu_sync = np.median(times_cpu_sync)

    # 2. CPU (ITK)
    print("\n--- 2. ITK Thinning (CPU) ---")
    itk_time = 0.0

    try:
        import itk

        print("Running ITK thinning (this may take a while)...")
        itk_image = itk.imread(input_path, itk.UC)
        start_time = time.time()
        thinning_filter = itk.BinaryThinningImageFilter3D.New(itk_image)
        thinning_filter.Update()
        itk_time = time.time() - start_time
        print(f"ITK Thinning finished in {itk_time:.4f} seconds.")
    except ImportError:
        print("itk module not found. Skipping CPU benchmark.")
        itk_time = None

    print("\n==============================================")
    print("               BENCHMARK SUMMARY                ")
    print("==============================================")
    print(f"Mode 0 (GPU Subgrid)             : {cuda_time_subgrid:.4f} s")
    print(f"Mode 1 (Hybrid CPU-Sync)         : {cuda_time_cpu_sync:.4f} s")
    if itk_time is not None:
        print(f"CPU (ITK)                        : {itk_time:.4f} s")
        print("----------------------------------------------")
        print(f"Speedup vs ITK (Mode 0)          : {itk_time / cuda_time_subgrid:.2f}x")
        print(
            f"Speedup vs ITK (Mode 1)          : {itk_time / cuda_time_cpu_sync:.2f}x"
        )
    print("==============================================")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "1_CT_HR_label_airways.nii.gz")
    run_benchmark(input_file, num_runs=5)
