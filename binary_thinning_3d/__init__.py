import torch
from . import cuda_thinning_ext


def binary_thinning(tensor: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """
    In-place 3D binary thinning on CUDA.

    Args:
        tensor (torch.Tensor): A 3D tensor on CUDA. All non-zero values are treated as foreground.
        mode (int):
            0 = GPU Subgrid (Fastest, preserves topology, fully GPU)
            1 = CPU Sequential Re-check (Matches ITK exactly, slower)

    Returns:
        torch.Tensor: The thinned binary tensor.
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA.")
    if tensor.dim() != 3:
        raise ValueError("Tensor must be 3D.")
    if mode not in [0, 1]:
        raise ValueError("Mode must be 0 (GPU Subgrid) or 1 (CPU Sequential).")

    # We must operate on a contiguous ByteTensor (uint8)
    if tensor.dtype != torch.uint8 or not tensor.is_contiguous():
        work_tensor = (tensor != 0).to(torch.uint8).contiguous()
    else:
        work_tensor = tensor

    cuda_thinning_ext.binary_thinning(work_tensor, mode)

    if work_tensor is not tensor:
        tensor.copy_(work_tensor)

    return tensor
