import torch
from . import cuda_thinning_ext

def binary_thinning(tensor: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    """
    In-place 3D binary thinning on CUDA.
    Args:
        tensor (torch.Tensor): A 3D tensor on CUDA. All non-zero values are treated as foreground.
        deterministic (bool): If True, forces the order of pixel deletions to be consistent and identical to CPU ITK. Slower.
    Returns:
        torch.Tensor: The thinned binary tensor.
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA.")
    if tensor.dim() != 3:
        raise ValueError("Tensor must be 3D.")
    
    # We must operate on a contiguous ByteTensor (uint8)
    if tensor.dtype != torch.uint8 or not tensor.is_contiguous():
        work_tensor = (tensor != 0).to(torch.uint8).contiguous()
    else:
        work_tensor = tensor

    cuda_thinning_ext.binary_thinning(work_tensor, deterministic)
    
    if work_tensor is not tensor:
        tensor.copy_(work_tensor)
    
    return tensor
