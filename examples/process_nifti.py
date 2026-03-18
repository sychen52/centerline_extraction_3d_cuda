import torch
import SimpleITK as sitk
import numpy as np
import time
from binary_thinning_3d import binary_thinning

def process_nifti(input_path, output_path):
    print(f"Loading {input_path}...")
    # Read the image
    img = sitk.ReadImage(input_path)
    # Convert to numpy array (Z, Y, X)
    img_array = sitk.GetArrayFromImage(img)
    
    print(f"Original shape: {img_array.shape}, type: {img_array.dtype}")
    print(f"Original sum (volume): {np.sum(img_array > 0)}")
    
    # Convert to PyTorch tensor on CUDA
    # Ensure it's uint8
    tensor = torch.from_numpy(img_array > 0).to(torch.uint8).cuda()
    
    print("Starting CUDA thinning...")
    start_time = time.time()
    
    # Run the thinning (in-place)
    binary_thinning(tensor)
    
    end_time = time.time()
    print(f"Thinning finished in {end_time - start_time:.4f} seconds.")
    
    # Transfer back to CPU and convert to numpy
    thinned_array = tensor.cpu().numpy()
    print(f"Thinned sum (volume): {np.sum(thinned_array)}")
    
    # Create a new SimpleITK image with the same metadata as the original
    thinned_img = sitk.GetImageFromArray(thinned_array)
    thinned_img.CopyInformation(img)
    
    print(f"Saving to {output_path}...")
    sitk.WriteImage(thinned_img, output_path)
    print("Done!")

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data', '1_CT_HR_label_airways.nii.gz')
    output_file = os.path.join(base_dir, 'data', '1_CT_HR_label_airways_thinned_cuda.nii.gz')
    process_nifti(input_file, output_file)
