import torch
import SimpleITK as sitk
import numpy as np
import time
import itk
import os
from binary_thinning_3d import binary_thinning

def process_nifti(input_path, output_path_cuda, output_path_itk):
    print(f"Loading {input_path}...")
    
    # -----------------------------------------
    # 1. CUDA Thinning (Deterministic)
    # -----------------------------------------
    img_sitk = sitk.ReadImage(input_path)
    img_array = sitk.GetArrayFromImage(img_sitk)
    
    print(f"Original shape: {img_array.shape}, type: {img_array.dtype}")
    print(f"Original sum (volume): {np.sum(img_array > 0)}")
    
    tensor = torch.from_numpy(img_array > 0).to(torch.uint8).cuda()
    
    print("\n--- Starting CUDA thinning (Deterministic) ---")
    start_time = time.time()
    binary_thinning(tensor, deterministic=True)
    cuda_time = time.time() - start_time
    print(f"CUDA Thinning finished in {cuda_time:.4f} seconds.")
    
    cuda_binary_array = (tensor.cpu().numpy() > 0).astype(np.uint8)
    
    # -----------------------------------------
    # 2. ITK Thinning (CPU)
    # -----------------------------------------
    print("\n--- Starting ITK thinning (CPU) ---")
    itk_image = itk.imread(input_path, itk.UC)
    
    start_time = time.time()
    thinning_filter = itk.BinaryThinningImageFilter3D.New(itk_image)
    thinning_filter.Update()
    itk_time = time.time() - start_time
    print(f"ITK Thinning finished in {itk_time:.4f} seconds.")
    
    itk_thinned_array = itk.array_from_image(thinning_filter.GetOutput())
    itk_binary_array = (itk_thinned_array > 0).astype(np.uint8)
    
    print(f"\nCUDA Thinned sum (volume): {np.sum(cuda_binary_array)}")
    print(f"ITK Thinned sum (volume): {np.sum(itk_binary_array)}")
    
    # -----------------------------------------
    # 3. Comparison
    # -----------------------------------------
    intersection = np.logical_and(cuda_binary_array, itk_binary_array).sum()
    union = np.logical_or(cuda_binary_array, itk_binary_array).sum()
    
    sum_cuda = np.sum(cuda_binary_array)
    sum_itk = np.sum(itk_binary_array)
    dice = 2.0 * intersection / (sum_cuda + sum_itk) if (sum_cuda + sum_itk) > 0 else 1.0
    diff = np.sum(cuda_binary_array != itk_binary_array)
    
    print("\n--- Comparison ---")
    print(f"Number of different pixels: {diff}")
    print(f"Intersection: {intersection}")
    print(f"Union: {union}")
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Are results identical? {'Yes' if diff == 0 else 'No'}")
    
    # -----------------------------------------
    # 4. Save Outputs
    # -----------------------------------------
    cuda_img = sitk.GetImageFromArray(cuda_binary_array)
    cuda_img.CopyInformation(img_sitk)
    sitk.WriteImage(cuda_img, output_path_cuda)
    
    itk_img = sitk.GetImageFromArray(itk_binary_array)
    itk_img.CopyInformation(img_sitk)
    sitk.WriteImage(itk_img, output_path_itk)

    print("\nDone!")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data', '1_CT_HR_label_airways.nii.gz')
    output_cuda = os.path.join(base_dir, 'data', '1_CT_HR_label_airways_thinned_cuda.nii.gz')
    output_itk = os.path.join(base_dir, 'data', '1_CT_HR_label_airways_thinned_itk.nii.gz')
    
    process_nifti(input_file, output_cuda, output_itk)
