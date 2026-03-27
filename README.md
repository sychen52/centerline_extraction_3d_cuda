# Binary Thinning 3D CUDA

This package provides a blazing fast, memory-efficient GPU implementation of 3D Binary Thinning (skeletonization) using CUDA and PyTorch. 

It is based on the [3D thinning algorithm by Lee and Kashyap (1994)](https://doi.org/10.1006/cvgi.1994.1039), which uses Euler characteristic invariance and 26-connectivity checks to safely erode a 3D binary volume down to a 1-pixel wide skeleton without altering its fundamental topology.

## Features

This implementation provides two topologically safe operating modes to suit your needs:

1. **Mode 0: GPU Subgrid 8-Color Parallel (`mode=0`, Default)**
   * **Speed:** Extremely Fast (~200x speedup over CPU)
   * **Behavior:** Operates entirely on the GPU. It avoids race conditions by partitioning the image into an 8-color 3D checkerboard. It re-checks and deletes pixels of the same color in parallel because they are mathematically guaranteed not to touch each other.
   * **Topology:** **Topologically Safe**. Produces a mathematically valid skeleton. *Note: Because the deletion order differs slightly from a strict CPU raster-scan, the exact pixel placement may differ very slightly from ITK (e.g. 0.003% difference), but the overall global topology is preserved perfectly.*
2. **Mode 1: Hybrid CPU-GPU Sequential (`mode=1`)**
   * **Speed:** Fast (~80x speedup over CPU)
   * **Behavior:** Calculates Euler invariance on the GPU in parallel, but performs the final 26-connectivity re-checks strictly sequentially on the CPU (using zero-overhead memory compaction). 
   * **Topology:** **100% Identical to ITK**. Guaranteed to produce the exact same pixel output as standard sequential CPU implementations like `itk.BinaryThinningImageFilter3D`.

## Installation

### Dependencies
* Python 3.8+
* PyTorch (with CUDA support)

```bash
git clone https://github.com/sychen52/binary_thinning_3d_cuda.git
cd binary_thinning_3d_cuda

# Standard install
pip install -e --no-build-isolation .

# Install with development dependencies (for running benchmarks)
pip install -e --no-build-isolation ".[dev]"
```
*(Note: `itk-thickness3d` and `SimpleITK` are **not** hard dependencies. They are only included in the `[dev]` extras for the purpose of benchmarking and validating against the CPU implementation).*

## Usage

The input must be a 3D contiguous PyTorch `uint8` (Byte) tensor located on a CUDA device. All non-zero values are treated as foreground (`0` for background, `>0` for foreground).

```python
import torch
from binary_thinning_3d import binary_thinning

# Create or load a 3D binary mask on the GPU
tensor = torch.zeros((100, 100, 100), dtype=torch.uint8, device='cuda')
tensor[25:75, 25:75, 25:75] = 1 # Solid block

# 1. GPU Subgrid (Default, Max Speed, Topologically Safe)
# Modifies the tensor in-place
binary_thinning(tensor, mode=0)

# 2. Hybrid CPU-GPU (Exact ITK Match)
binary_thinning(tensor, mode=1)
```

## Benchmark

The following benchmark was run on a `(767, 512, 512)` NIfTI volume (CT Airways Label) containing `451,530` foreground voxels. 

The benchmark compares this CUDA implementation against `itk.BinaryThinningImageFilter3D` (which is run sequentially on the CPU).

| Method | Output Voxel Count | Time (Seconds) | Speedup vs ITK | Matches ITK CPU? |
| :--- | :--- | :--- | :--- | :--- |
| **Mode 0 (GPU Subgrid)** | 4,286 | **0.72 s** | **194x** | Topologically equivalent |
| **Mode 1 (Hybrid CPU)** | 4,281 | 1.82 s | 77x | **Yes (100% Identical)** |
| **ITK (CPU Baseline)** | 4,281 | 140.27 s | 1x | Baseline |

To reproduce these benchmarks yourself:
```bash
# Ensure you installed with dev dependencies: pip install -e ".[dev]"
python examples/process_nifti.py
```
*(The script will cache the slow ITK result to disk on the first run, so subsequent runs finish instantly).*
