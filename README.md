# Binary Thinning 3D CUDA

This package provides a blazing fast, memory-efficient GPU implementation of 3D Binary Thinning (skeletonization) using CUDA and PyTorch. 

It is based on the [3D thinning algorithm by Lee and Kashyap (1994)](https://doi.org/10.1006/cvgi.1994.1039), which uses Euler characteristic invariance and 26-connectivity checks to safely erode a 3D binary volume down to a 1-pixel wide skeleton without altering its fundamental topology.

## Features

This implementation provides two operating modes to suit your needs:

1. **Non-Deterministic (Fastest):** Operates entirely on the GPU in parallel. It is aggressively fast, reaching up to ~190x speedup over standard CPU implementations. However, due to parallel race conditions during connectivity checks, the final skeleton may differ slightly from a strictly sequential CPU run.
2. **Deterministic (Identical to CPU):** A hybrid CPU-GPU approach. The heavy mathematical filtering (Euler checks) happens in parallel on the GPU, while the connectivity re-checking is safely managed sequentially by the CPU using zero-overhead memory compaction. It produces a 100% pixel-perfect identical result to the standard ITK CPU implementation while still maintaining an ~85x speedup.

## Installation

### Dependencies
* Python 3.8+
* PyTorch (with CUDA support)

```bash
git clone <repository_url>
cd binary_thinning_3d_cuda

# Standard install
pip install -e .

# Install with development dependencies (for running benchmarks)
pip install -e ".[dev]"
```
*(Note: `itk-thickness3d` and `SimpleITK` are **not** hard dependencies. They are only included in the `[dev]` extras for the purpose of benchmarking and validating against the CPU implementation).*

## Usage

The input must be a 3D contiguous PyTorch `uint8` (Byte) tensor located on a CUDA device. Values should be purely binary (`0` for background, `1` for foreground).

```python
import torch
from binary_thinning_3d import binary_thinning

# Create or load a 3D binary mask on the GPU
# Example: 100x100x100 volume
tensor = torch.zeros((100, 100, 100), dtype=torch.uint8, device='cuda')
tensor[25:75, 25:75, 25:75] = 1 # Solid block

# 1. Non-Deterministic (Max Speed)
# Modifies the tensor in-place
binary_thinning(tensor, deterministic=False)

# 2. Deterministic (Exact ITK Match)
binary_thinning(tensor, deterministic=True)
```

## Benchmark

The following benchmark was run on a `(767, 512, 512)` NIfTI volume (CT Airways Label) containing `451,530` foreground voxels. 

The benchmark compares this CUDA implementation against `itk.BinaryThinningImageFilter3D` (which is run sequentially on the CPU).

| Method | Output Voxel Count | Time (Seconds) | Speedup vs ITK | Matches ITK CPU? |
| :--- | :--- | :--- | :--- | :--- |
| **CUDA (Non-Deterministic)** | 2,441 | **0.71 s** | **196x** | No (Slightly aggressive) |
| **CUDA (Deterministic)** | 4,281 | **1.65 s** | **84x** | **Yes (100% Identical)** |
| **ITK (CPU)** | 4,281 | 140.27 s | 1x | Baseline |

To reproduce these benchmarks yourself:
```bash
# Ensure you installed with dev dependencies: pip install -e ".[dev]"
python examples/process_nifti.py
```
