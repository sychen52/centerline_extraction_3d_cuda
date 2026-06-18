import os
import torch
import numpy as np
import SimpleITK as sitk
from centerline_extraction_3d_cuda import binary_thinning


def test_binary_thinning_hybrid_matches_itk():
    """
    Tests that the mode 1 (Hybrid CPU Sync) binary_thinning exactly matches ITK's CPU thinning
    to guarantee topological correctness.

    The ground-truth output was computed from the CPU version (ITK).
    We load these pre-computed NIfTI images directly from the data/ folder.
    SimpleITK is listed in dev dependencies so it is available in the test environment.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "1_CT_HR_label_airways.nii.gz")
    itk_output_path = os.path.join(
        base_dir, "data", "1_CT_HR_label_airways_thinned_itk.nii.gz"
    )

    # Load data using SimpleITK
    input_img = sitk.ReadImage(input_path)
    input_array = sitk.GetArrayFromImage(input_img)

    itk_output_img = sitk.ReadImage(itk_output_path)
    itk_output_array = sitk.GetArrayFromImage(itk_output_img)

    # Convert to torch GPU tensor (ensure it is uint8)
    tensor_input = torch.from_numpy(input_array > 0).to(torch.uint8).cuda()

    # Run hybrid CUDA thinning (mode=1 guarantees exact ITK sequential match)
    output_tensor = binary_thinning(tensor_input, mode=1)

    # Convert back to numpy for comparison
    cuda_output_array = output_tensor.cpu().numpy()

    # Compare
    diff = np.sum(cuda_output_array != itk_output_array)

    assert diff == 0, f"CUDA Mode 1 output differed from ITK by {diff} voxels!"


if __name__ == "__main__":
    test_binary_thinning_hybrid_matches_itk()
    print("All tests passed.")
