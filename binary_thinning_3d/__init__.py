import torch
from . import cuda_thinning_ext


class ExtractCenterlineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, probs, mode):
        # We clone mask to keep the original for backward pass
        # Since the C++ function modifies it in-place
        cl = mask.clone()
        if cl.dtype != torch.uint8:
            cl = cl.to(torch.uint8)
        if not cl.is_contiguous():
            cl = cl.contiguous()

        cl.clamp_(0, 1)

        # Clone probs to avoid modifying the user's tensor if it requires grad
        cl_probs = probs.clone()
        if cl_probs.dtype != torch.float32:
            cl_probs = cl_probs.to(torch.float32)
        if not cl_probs.is_contiguous():
            cl_probs = cl_probs.contiguous()

        # Run forward pass (modifies cl and cl_probs in-place)
        cuda_thinning_ext.extract_centerline(cl, cl_probs, mode)

        # Save for backward pass
        # We need the original mask, and the resulting skeleton
        ctx.save_for_backward(mask, cl)
        ctx.mode = mode

        return cl_probs

    @staticmethod
    def backward(ctx, grad_output):
        mask, cl = ctx.saved_tensors

        if grad_output is None:
            return None, None, None

        # The mask must be uint8
        if mask.dtype != torch.uint8:
            mask = mask.to(torch.uint8)
        if not mask.is_contiguous():
            mask = mask.contiguous()

        grad_prob = torch.zeros_like(grad_output)

        # Run backward pass region growing
        cuda_thinning_ext.region_grow_backward(
            mask, cl, grad_output.contiguous(), grad_prob
        )

        # Return gradients for (mask, probs, mode)
        # mask is discrete, mode is int, so only probs gets a gradient
        return None, grad_prob, None


def binary_thinning(tensor: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """
    3D binary thinning using CUDA.
    The operation is performed on the tensor provided (in-place for the binary representation).
    If the tensor is on CPU, it will be moved to CUDA for processing and then copied back to the original tensor.

    Args:
        tensor (torch.Tensor): A 3D tensor. All non-zero values are treated as foreground.
        mode (int):
            0 = GPU Subgrid (Fastest, preserves topology, fully GPU)
            1 = CPU Sequential Re-check (Matches ITK exactly, slower)

    Returns:
        torch.Tensor: The thinned binary tensor.
    """
    if tensor.dim() != 3:
        raise ValueError("Tensor must be 3D.")
    if mode not in [0, 1]:
        raise ValueError("Mode must be 0 (GPU Subgrid) or 1 (CPU Sequential).")

    # Ensure it's uint8 and contiguous before passing to CUDA extension
    # We do this check here to keep the C++ extension focused on the algorithm
    if tensor.dtype != torch.uint8 or not tensor.is_contiguous():
        # This creates a new tensor, we won't be able to modify the original in-place
        # if the user passed something like a float tensor or a non-contiguous one.
        work_tensor = (tensor != 0).to(torch.uint8).contiguous()
        cuda_thinning_ext.binary_thinning(work_tensor, mode)
        # If the original was on the same device, we can try to copy back
        if tensor.shape == work_tensor.shape:
            try:
                tensor.copy_(work_tensor)
            except Exception:
                pass  # Might fail if types are incompatible for copy_
        return work_tensor
    else:
        # In-place ensure binary (0 or 1)
        tensor.clamp_(0, 1)
        cuda_thinning_ext.binary_thinning(tensor, mode)
        return tensor


def extract_centerline(
    mask: torch.Tensor, probs: torch.Tensor, mode: int = 0
) -> torch.Tensor:
    """
    Differentiable 3D thinning using CUDA.

    Performs an inward max-pooling sweep during the thinning to propagate continuous values to the skeleton.
    Supports PyTorch Autograd! The output skeleton will have requires_grad=True if probs has requires_grad=True.
    The backward pass uses dense unweighted region-growing (Voronoi partitioning).

    Args:
        mask (torch.Tensor): A 3D binary tensor defining the topology.
        probs (torch.Tensor): A 3D float32 continuous probability map.
        mode (int):
            0 = GPU Subgrid (Fastest, preserves topology, fully GPU)
            1 = CPU Sequential Re-check (Matches ITK exactly, slower)

    Returns:
        torch.Tensor: The continuous skeleton probabilities.
    """
    if mask.dim() != 3 or probs.dim() != 3:
        raise ValueError("Tensors must be 3D.")
    if mode not in [0, 1]:
        raise ValueError("Mode must be 0 (GPU Subgrid) or 1 (CPU Sequential).")

    return ExtractCenterlineFunction.apply(mask, probs, mode)
