# Binary Thinning 3D CUDA

This package provides a blazing fast, memory-efficient GPU implementation of 3D Binary Thinning (skeletonization) using CUDA and PyTorch. 

It is based on the [3D thinning algorithm by Lee, Kashyap and Chu (1994)](https://doi.org/10.1006/cgip.1994.1042), which uses Euler characteristic invariance and 26-connectivity checks to safely erode a 3D binary volume down to a 1-pixel wide skeleton without altering its fundamental topology.

## Features

This implementation provides two topologically safe operating modes to suit your needs:

1. **Mode 0: GPU Subgrid 8-Color Parallel (`mode=0`, Default)**
   * **Speed:** Extremely Fast (~300x speedup over CPU)
   * **Behavior:** Operates entirely on the GPU. It avoids race conditions by partitioning the image into an 8-color 3D checkerboard. It re-checks and deletes pixels of the same color in parallel because they are mathematically guaranteed not to touch each other.
   * **Topology:** **Topologically Safe**. Produces a mathematically valid skeleton. *Note: Because the deletion order differs slightly from a strict CPU raster-scan, the exact pixel placement may differ very slightly from ITK (e.g. 0.003% difference), but the overall global topology is preserved perfectly.*
2. **Mode 1: Hybrid CPU-GPU Sequential (`mode=1`)**
   * **Speed:** Fast (~100x speedup over CPU)
   * **Behavior:** Calculates Euler invariance on the GPU in parallel, but performs the final 26-connectivity re-checks strictly sequentially on the CPU (using zero-overhead memory compaction and host-side sorting). 
   * **Topology:** **100% Identical to ITK**. Guaranteed to produce the exact same pixel output as standard sequential CPU implementations like `itk.BinaryThinningImageFilter3D`.

## Installation

### Prerequisites
* Python 3.10+
* PyTorch (with CUDA support)
* A CUDA-capable GPU

### Install from PyPI (Recommended)
You can install the package directly from PyPI. Note that since this contains CUDA C++ extensions, it will be compiled on your machine during installation.
```bash
pip install binary-thinning-3d-cuda
```

### Install from Source (Advanced Users)
For development or to run benchmarks, you can install from the source:
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

The input can be a 3D PyTorch `uint8` (Byte) tensor located on either a **CPU or CUDA device**. 

* If the tensor is on a **CUDA device**, the operation is performed in-place.
* If the tensor is on the **CPU**, it is automatically moved to the GPU for processing and copied back to the original CPU tensor in-place.

All non-zero values are treated as foreground (`0` for background, `>0` for foreground).

```python
import torch
from binary_thinning_3d import binary_thinning

# Create or load a 3D binary mask (CPU or GPU)
tensor = torch.zeros((100, 100, 100), dtype=torch.uint8)
tensor[25:75, 25:75, 25:75] = 1 # Solid block

# 1. GPU Subgrid (Default, Max Speed, Topologically Safe)
# Modifies the tensor in-place (handles CPU<->GPU transfer automatically)
binary_thinning(tensor, mode=0)

# 2. Hybrid CPU-GPU (Exact ITK Match)
binary_thinning(tensor, mode=1)
```

## Benchmark

The following benchmark was run on a `(767, 512, 512)` NIfTI volume (CT Airways Label) containing `451,530` foreground voxels. 

The benchmark compares this CUDA implementation against `itk.BinaryThinningImageFilter3D` (which is run sequentially on the CPU). The CUDA timings **include** the time for CPU-to-GPU and GPU-to-CPU data transfers.

| Method | Output Voxel Count | Time (Seconds) | Speedup vs ITK | Matches ITK CPU? |
| :--- | :--- | :--- | :--- | :--- |
| **Mode 0 (GPU Subgrid)** | 4,286 | **0.38 s** | **331x** | Topologically equivalent |
| **Mode 1 (Hybrid CPU)** | 4,281 | 1.22 s | 101x | **Yes (100% Identical)** |
| **ITK (CPU Baseline)** | 4,281 | 139.90 s | 1x | Baseline |

To reproduce these benchmarks yourself:
```bash
# Ensure you installed with dev dependencies: pip install -e ".[dev]"
python examples/process_nifti.py
```
*(The script will cache the slow ITK result to disk on the first run, so subsequent runs finish instantly).*
