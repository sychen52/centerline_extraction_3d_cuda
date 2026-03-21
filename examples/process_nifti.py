import torch
import SimpleITK as sitk
import numpy as np
import time
import os
from binary_thinning_3d import binary_thinning

def process_nifti(input_path):
    print(f"Loading {input_path}...")
    
    img_sitk = sitk.ReadImage(input_path)
    img_array = sitk.GetArrayFromImage(img_sitk)
    
    print(f"Original shape: {img_array.shape}, type: {img_array.dtype}")
    print(f"Original sum (volume): {np.sum(img_array > 0)}")
    
    tensor_non_det = torch.from_numpy(img_array > 0).to(torch.uint8).cuda()
    tensor_det = torch.from_numpy(img_array > 0).to(torch.uint8).cuda()
    
    # 1. GPU Non-Deterministic
    print("\n--- 1. Starting CUDA thinning (Non-Deterministic) ---")
    torch.cuda.synchronize()
    start_time = time.time()
    binary_thinning(tensor_non_det, deterministic=False)
    torch.cuda.synchronize()
    cuda_time_non_det = time.time() - start_time
    print(f"CUDA Non-Deterministic Thinning finished in {cuda_time_non_det:.4f} seconds.")
    cuda_binary_array_non_det = (tensor_non_det.cpu().numpy() > 0).astype(np.uint8)
    
    # 2. GPU Deterministic
    print("\n--- 2. Starting CUDA thinning (Deterministic) ---")
    torch.cuda.synchronize()
    start_time = time.time()
    binary_thinning(tensor_det, deterministic=True)
    torch.cuda.synchronize()
    cuda_time_det = time.time() - start_time
    print(f"CUDA Deterministic Thinning finished in {cuda_time_det:.4f} seconds.")
    cuda_binary_array_det = (tensor_det.cpu().numpy() > 0).astype(np.uint8)
    
    # 3. CPU (ITK)
    print("\n--- 3. Starting ITK thinning (CPU) ---")
    # Import inside the block so it's clear it's optional
    try:
        import itk
    except ImportError:
        print("itk module not found. Skipping CPU benchmark.")
        itk_binary_array = None
    else:
        itk_image = itk.imread(input_path, itk.UC)
        start_time = time.time()
        thinning_filter = itk.BinaryThinningImageFilter3D.New(itk_image)
        thinning_filter.Update()
        itk_time = time.time() - start_time
        print(f"ITK Thinning finished in {itk_time:.4f} seconds.")
        itk_thinned_array = itk.array_from_image(thinning_filter.GetOutput())
        itk_binary_array = (itk_thinned_array > 0).astype(np.uint8)

    print("\n--- Summary ---")
    print(f"Original sum (volume)            : {np.sum(img_array > 0)}")
    print(f"CUDA Non-Deterministic sum       : {np.sum(cuda_binary_array_non_det)}")
    print(f"CUDA Deterministic sum           : {np.sum(cuda_binary_array_det)}")
    if itk_binary_array is not None:
        print(f"ITK CPU sum                      : {np.sum(itk_binary_array)}")
    
    print("\n--- Timing ---")
    print(f"1. GPU Non-Deterministic         : {cuda_time_non_det:.4f} s")
    print(f"2. GPU Deterministic             : {cuda_time_det:.4f} s")
    if itk_binary_array is not None:
        print(f"3. CPU (ITK)                     : {itk_time:.4f} s")
        speedup_non_det = itk_time / cuda_time_non_det
        speedup_det = itk_time / cuda_time_det
        print(f"Speedup vs ITK (Non-Det)         : {speedup_non_det:.2f}x")
        print(f"Speedup vs ITK (Det)             : {speedup_det:.2f}x")

    if itk_binary_array is not None:
        print("\n--- Comparison (GPU Deterministic vs CPU ITK) ---")
        diff = np.sum(cuda_binary_array_det != itk_binary_array)
        print(f"Number of different pixels: {diff}")
        print(f"Are results identical? {'Yes' if diff == 0 else 'No'}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data', '1_CT_HR_label_airways.nii.gz')
    process_nifti(input_file)
