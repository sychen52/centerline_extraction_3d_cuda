import torch
import SimpleITK as sitk
import numpy as np
import time
import os
from binary_thinning_3d import binary_thinning


def process_nifti(input_path, num_runs=5):
    print(f"Loading {input_path}...")

    img_sitk = sitk.ReadImage(input_path)
    img_array = sitk.GetArrayFromImage(img_sitk)

    print(f"Original shape: {img_array.shape}, type: {img_array.dtype}")
    print(f"Original sum (volume): {np.sum(img_array > 0)}")

    # Prepare base tensor on GPU
    base_tensor = torch.from_numpy(img_array > 0).to(torch.uint8).cuda()

    # 0. GPU Subgrid (8-color)
    print(
        f"\n--- 0. Starting CUDA thinning (GPU Subgrid 8-Color) - Running {num_runs} times ---"
    )
    times_subgrid = []
    cuda_binary_array_subgrid = None

    for i in range(num_runs):
        tensor_subgrid = base_tensor.clone()
        torch.cuda.synchronize()
        start_time = time.time()
        binary_thinning(tensor_subgrid, mode=0)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        times_subgrid.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.4f}s")
        if i == num_runs - 1:
            cuda_binary_array_subgrid = (tensor_subgrid.cpu().numpy() > 0).astype(
                np.uint8
            )

    cuda_time_subgrid = np.median(times_subgrid)
    print(f"Median CUDA GPU Subgrid Thinning: {cuda_time_subgrid:.4f} seconds.")

    # 1. GPU Hybrid (CPU Sync)
    print(
        f"\n--- 1. Starting CUDA thinning (Hybrid CPU-Sync) - Running {num_runs} times ---"
    )
    times_cpu_sync = []
    cuda_binary_array_cpu_sync = None

    for i in range(num_runs):
        tensor_cpu_sync = base_tensor.clone()
        torch.cuda.synchronize()
        start_time = time.time()
        binary_thinning(tensor_cpu_sync, mode=1)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        times_cpu_sync.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.4f}s")
        if i == num_runs - 1:
            cuda_binary_array_cpu_sync = (tensor_cpu_sync.cpu().numpy() > 0).astype(
                np.uint8
            )

    cuda_time_cpu_sync = np.median(times_cpu_sync)
    print(f"Median CUDA Hybrid CPU-Sync Thinning: {cuda_time_cpu_sync:.4f} seconds.")

    # 2. CPU (ITK)
    print("\n--- 2. Starting ITK thinning (CPU) ---")
    itk_binary_array = None
    itk_time = 0.0

    cache_prefix = input_path.replace(".nii.gz", "").replace(".nii", "")
    itk_cache_path = f"{cache_prefix}_itk_result.npy"
    itk_time_cache_path = f"{cache_prefix}_itk_time.txt"

    if os.path.exists(itk_cache_path) and os.path.exists(itk_time_cache_path):
        print(f"Loading cached ITK result from {itk_cache_path}...")
        itk_binary_array = np.load(itk_cache_path)
        with open(itk_time_cache_path, "r") as f:
            itk_time = float(f.read())
        print(f"Loaded cached ITK time: {itk_time:.4f}s")
    else:
        try:
            import itk

            print("Running ITK thinning (this may take a while)...")
            itk_image = itk.imread(input_path, itk.UC)
            start_time = time.time()
            thinning_filter = itk.BinaryThinningImageFilter3D.New(itk_image)
            thinning_filter.Update()
            itk_time = time.time() - start_time
            print(f"ITK Thinning finished in {itk_time:.4f} seconds.")

            itk_thinned_array = itk.array_from_image(thinning_filter.GetOutput())
            itk_binary_array = (itk_thinned_array > 0).astype(np.uint8)

            # Save to cache
            np.save(itk_cache_path, itk_binary_array)
            with open(itk_time_cache_path, "w") as f:
                f.write(str(itk_time))
            print("Saved ITK result and timing to cache.")
        except ImportError:
            print("itk module not found. Skipping CPU benchmark.")
        except Exception as e:
            print(f"Error running ITK thinning: {e}")

    print("\n--- Summary ---")
    print(f"Original sum (volume)            : {np.sum(img_array > 0)}")
    print(f"Mode 0 (GPU Subgrid) sum         : {np.sum(cuda_binary_array_subgrid)}")
    print(f"Mode 1 (Hybrid CPU) sum          : {np.sum(cuda_binary_array_cpu_sync)}")
    if itk_binary_array is not None:
        print(f"ITK CPU sum                      : {np.sum(itk_binary_array)}")

    print("\n--- Timing (Median of GPU runs) ---")
    print(f"Mode 0 (GPU Subgrid)             : {cuda_time_subgrid:.4f} s")
    print(f"Mode 1 (Hybrid CPU)              : {cuda_time_cpu_sync:.4f} s")
    if itk_binary_array is not None:
        print(f"2. CPU (ITK)                     : {itk_time:.4f} s")
        speedup_subgrid = itk_time / cuda_time_subgrid
        speedup_cpu_sync = itk_time / cuda_time_cpu_sync
        print(f"Speedup vs ITK (Mode 0)          : {speedup_subgrid:.2f}x")
        print(f"Speedup vs ITK (Mode 1)          : {speedup_cpu_sync:.2f}x")

    if itk_binary_array is not None:
        print("\n--- Comparison (GPU vs CPU ITK) ---")
        diff_0 = np.sum(cuda_binary_array_subgrid != itk_binary_array)
        diff_1 = np.sum(cuda_binary_array_cpu_sync != itk_binary_array)
        print(
            f"Mode 0 differences from ITK      : {diff_0} (due to parallel subgrid deletion order)"
        )
        print(f"Mode 1 differences from ITK      : {diff_1}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "1_CT_HR_label_airways.nii.gz")
    process_nifti(input_file, num_runs=5)
