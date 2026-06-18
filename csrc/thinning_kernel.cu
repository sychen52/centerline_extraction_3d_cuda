/**
 * 3D Binary Thinning CUDA Kernel
 *
 * Voxel value assumptions:
 * - 0: Background
 * - 1: Foreground (object)
 * - 2: Internal marker for "candidate for deletion" during a thinning
 * iteration.
 *
 * The algorithm iteratively identifies border points that are "simple" (can be
 * removed without changing the topology) and marks them with 2. It then
 * resolves these candidates either sequentially (Mode 1) or in parallel
 * subgrids (Mode 2) to ensure topological correctness.
 */
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optional>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <torch/extension.h>
#include <vector>

__constant__ int d_eulerLUT[256] = {
    0, 1,  0, -1, 0, -1, 0, 1,  0, -3, 0, -1, 0, -1, 0, 1,  0, -1, 0, 1,  0, 1,
    0, -1, 0, 3,  0, 1,  0, 1,  0, -1, 0, -3, 0, -1, 0, 3,  0, 1,  0, 1,  0, -1,
    0, 3,  0, 1,  0, -1, 0, 1,  0, 1,  0, -1, 0, 3,  0, 1,  0, 1,  0, -1, 0, -3,
    0, 3,  0, -1, 0, 1,  0, 1,  0, 3,  0, -1, 0, 1,  0, -1, 0, 1,  0, 1,  0, -1,
    0, 3,  0, 1,  0, 1,  0, -1, 0, 1,  0, 3,  0, 3,  0, 1,  0, 5,  0, 3,  0, 3,
    0, 1,  0, -1, 0, 1,  0, 1,  0, -1, 0, 3,  0, 1,  0, 1,  0, -1, 0, -7, 0, -1,
    0, -1, 0, 1,  0, -3, 0, -1, 0, -1, 0, 1,  0, -1, 0, 1,  0, 1,  0, -1, 0, 3,
    0, 1,  0, 1,  0, -1, 0, -3, 0, -1, 0, 3,  0, 1,  0, 1,  0, -1, 0, 3,  0, 1,
    0, -1, 0, 1,  0, 1,  0, -1, 0, 3,  0, 1,  0, 1,  0, -1, 0, -3, 0, 3,  0, -1,
    0, 1,  0, 1,  0, 3,  0, -1, 0, 1,  0, -1, 0, 1,  0, 1,  0, -1, 0, 3,  0, 1,
    0, 1,  0, -1, 0, 1,  0, 3,  0, 3,  0, 1,  0, 5,  0, 3,  0, 3,  0, 1,  0, -1,
    0, 1,  0, 1,  0, -1, 0, 3,  0, 1,  0, 1,  0, -1};

__device__ bool is_euler_invariant(const int *neighbors) {
  int eulerChar = 0;
  unsigned char n;

  // Octant SWU
  n = 1;
  if (neighbors[24] == 1)
    n |= 128;
  if (neighbors[25] == 1)
    n |= 64;
  if (neighbors[15] == 1)
    n |= 32;
  if (neighbors[16] == 1)
    n |= 16;
  if (neighbors[21] == 1)
    n |= 8;
  if (neighbors[22] == 1)
    n |= 4;
  if (neighbors[12] == 1)
    n |= 2;
  eulerChar += d_eulerLUT[n];

  // Octant SEU
  n = 1;
  if (neighbors[26] == 1)
    n |= 128;
  if (neighbors[23] == 1)
    n |= 64;
  if (neighbors[17] == 1)
    n |= 32;
  if (neighbors[14] == 1)
    n |= 16;
  if (neighbors[25] == 1)
    n |= 8;
  if (neighbors[22] == 1)
    n |= 4;
  if (neighbors[16] == 1)
    n |= 2;
  eulerChar += d_eulerLUT[n];

  // Octant NWU
  n = 1;
  if (neighbors[18] == 1)
    n |= 128;
  if (neighbors[21] == 1)
    n |= 64;
  if (neighbors[9] == 1)
    n |= 32;
  if (neighbors[12] == 1)
    n |= 16;
  if (neighbors[19] == 1)
    n |= 8;
  if (neighbors[22] == 1)
    n |= 4;
  if (neighbors[10] == 1)
    n |= 2;
  eulerChar += d_eulerLUT[n];

  // Octant NEU
  n = 1;
  if (neighbors[20] == 1)
    n |= 128;
  if (neighbors[23] == 1)
    n |= 64;
  if (neighbors[19] == 1)
    n |= 32;
  if (neighbors[22] == 1)
    n |= 16;
  if (neighbors[11] == 1)
    n |= 8;
  if (neighbors[14] == 1)
    n |= 4;
  if (neighbors[10] == 1)
    n |= 2;
  eulerChar += d_eulerLUT[n];

  // Octant SWB
  n = 1;
  if (neighbors[6] == 1)
    n |= 128;
  if (neighbors[15] == 1)
    n |= 64;
  if (neighbors[7] == 1)
    n |= 32;
  if (neighbors[16] == 1)
    n |= 16;
  if (neighbors[3] == 1)
    n |= 8;
  if (neighbors[12] == 1)
    n |= 4;
  if (neighbors[4] == 1)
    n |= 2;
  eulerChar += d_eulerLUT[n];

  // Octant SEB
  n = 1;
  if (neighbors[8] == 1)
    n |= 128;
  if (neighbors[7] == 1)
    n |= 64;
  if (neighbors[17] == 1)
    n |= 32;
  if (neighbors[16] == 1)
    n |= 16;
  if (neighbors[5] == 1)
    n |= 8;
  if (neighbors[4] == 1)
    n |= 4;
  if (neighbors[14] == 1)
    n |= 2;
  eulerChar += d_eulerLUT[n];

  // Octant NWB
  n = 1;
  if (neighbors[0] == 1)
    n |= 128;
  if (neighbors[9] == 1)
    n |= 64;
  if (neighbors[3] == 1)
    n |= 32;
  if (neighbors[12] == 1)
    n |= 16;
  if (neighbors[1] == 1)
    n |= 8;
  if (neighbors[10] == 1)
    n |= 4;
  if (neighbors[4] == 1)
    n |= 2;
  eulerChar += d_eulerLUT[n];

  // Octant NEB
  n = 1;
  if (neighbors[2] == 1)
    n |= 128;
  if (neighbors[1] == 1)
    n |= 64;
  if (neighbors[11] == 1)
    n |= 32;
  if (neighbors[10] == 1)
    n |= 16;
  if (neighbors[5] == 1)
    n |= 8;
  if (neighbors[4] == 1)
    n |= 4;
  if (neighbors[14] == 1)
    n |= 2;
  eulerChar += d_eulerLUT[n];

  return (eulerChar == 0);
}

__host__ __device__ void get_neighbors(const unsigned char *img, int d, int h,
                                       int w, int x, int y, int z,
                                       int *neighbors) {
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;
        int n_idx = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);

        int val = 0;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h && nz >= 0 && nz < d) {
          size_t flat_n_idx = (size_t)nz * (h * w) + ny * w + nx;
          val = img[flat_n_idx];
        }
        neighbors[n_idx] = (val > 0) ? 1 : 0;
      }
    }
  }
}

/**
 * Using region growing (similar to Octree labeling).
 */
__host__ __device__ bool is_simple_point(const int *neighbors) {
  int temp_neighbors[27]; // 0: background; 1: unvisited foreground; 2: visited
                          // foreground
  for (int i = 0; i < 27; ++i) {
    temp_neighbors[i] = neighbors[i];
  }

  int components = 0;
  for (int i = 0; i < 27; ++i) {
    if (i == 13 || temp_neighbors[i] != 1)
      continue;

    // Found a new component
    components++;
    if (components > 1)
      return false;

    // Region growing to mark all voxels in this component
    int stack[27];
    int stack_ptr = 0;
    stack[stack_ptr++] = i;
    temp_neighbors[i] = 2; // Mark as visited

    while (stack_ptr > 0) {
      int curr = stack[--stack_ptr];
      int cx = curr % 3;
      int cy = (curr / 3) % 3;
      int cz = curr / 9;

      // Check all 26-neighbors of 'curr' WITHIN the 3x3x3 window
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0 && dz == 0)
              continue;
            int nx = cx + dx;
            int ny = cy + dy;
            int nz = cz + dz;

            if (nx >= 0 && nx < 3 && ny >= 0 && ny < 3 && nz >= 0 && nz < 3) {
              int n_idx = nz * 9 + ny * 3 + nx;
              if (n_idx != 13 && temp_neighbors[n_idx] == 1) {
                temp_neighbors[n_idx] = 2;
                stack[stack_ptr++] = n_idx;
              }
            }
          }
        }
      }
    }
  }

  return (components == 1);
}

__global__ void mark_deletable_points_kernel(unsigned char *img, int d, int h,
                                             int w, int currentBorder,
                                             unsigned int *marked_indices,
                                             int *marked_count) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= w || y >= h || z >= d)
    return;

  size_t idx = (size_t)z * (h * w) + y * w + x;
  if (img[idx] == 0 || img[idx] == 2)
    return;

  int neighbors[27];
  get_neighbors(img, d, h, w, x, y, z, neighbors);

  bool isBorderPoint = false;
  if (currentBorder == 1 && neighbors[10] == 0)
    isBorderPoint = true; // N
  else if (currentBorder == 2 && neighbors[16] == 0)
    isBorderPoint = true; // S
  else if (currentBorder == 3 && neighbors[14] == 0)
    isBorderPoint = true; // E
  else if (currentBorder == 4 && neighbors[12] == 0)
    isBorderPoint = true; // W
  else if (currentBorder == 5 && neighbors[22] == 0)
    isBorderPoint = true; // U
  else if (currentBorder == 6 && neighbors[4] == 0)
    isBorderPoint = true; // B

  if (!isBorderPoint)
    return;

  int num_neighbors = 0;
  for (int i = 0; i < 27; ++i) {
    if (i != 13 && neighbors[i] == 1)
      num_neighbors++;
  }

  if (num_neighbors == 1)
    return;
  if (!is_euler_invariant(neighbors))
    return;
  if (!is_simple_point(neighbors))
    return;

  img[idx] = 2; // mark for deletion
  int pos = atomicAdd(marked_count, 1);
  marked_indices[pos] = (unsigned int)idx;
}

__global__ void subgrid_recheck_kernel(unsigned char *img, int d, int h, int w,
                                       unsigned int *marked_indices, int count,
                                       int color, int *changed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;

  size_t idx = marked_indices[i];
  if (img[idx] != 2)
    return; // Already handled by previous color or untouched

  int x = idx % w;
  int y = (idx / w) % h;
  int z = idx / (w * h);

  int p = (x % 2) + (y % 2) * 2 + (z % 2) * 4;
  if (p != color)
    return;

  img[idx] = 0; // Temporarily delete

  int neighbors[27];
  get_neighbors(img, d, h, w, x, y, z, neighbors);

  if (!is_simple_point(neighbors)) {
    img[idx] = 1; // Not simple anymore, restore
  } else {
    atomicAdd(changed, 1);
  }
}

__global__ void apply_updates_kernel(unsigned char *img,
                                     const unsigned int *indices,
                                     const unsigned char *values, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    img[indices[i]] = values[i];
  }
}

__global__ void inward_sweep_kernel(const unsigned char *img,
                                    const float *probs_in, float *probs_out,
                                    int d, int h, int w) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= w || y >= h || z >= d)
    return;

  size_t idx = (size_t)z * (h * w) + y * w + x;

  if (img[idx] == 0) {
    probs_out[idx] = 0.0f;
    return;
  }

  float max_p = probs_in[idx];

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;
        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h && nz >= 0 && nz < d) {
          size_t n_idx = (size_t)nz * (h * w) + ny * w + nx;
          float p = probs_in[n_idx];
          if (p > max_p) {
            max_p = p;
          }
        }
      }
    }
  }
  probs_out[idx] = max_p;
}

void thinning_internal(torch::Tensor image, int mode,
                       std::optional<torch::Tensor> probs) {
  TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
  TORCH_CHECK(image.scalar_type() == torch::kByte,
              "image must be a ByteTensor (uint8)");
  TORCH_CHECK(image.dim() == 3, "image must be a 3D tensor");

  int d = image.size(0);
  int h = image.size(1);
  int w = image.size(2);
  size_t total_size = (size_t)d * h * w;

  bool is_cpu = !image.is_cuda();
  torch::Tensor d_tensor;
  if (is_cpu) {
    d_tensor = image.to(torch::kCUDA);
  } else {
    d_tensor = image;
  }
  unsigned char *d_img = d_tensor.data_ptr<unsigned char>();

  bool use_probs = probs.has_value();
  float *d_probs_in = nullptr;
  float *d_probs_out = nullptr;
  torch::Tensor d_probs_tensor;
  torch::Tensor d_probs_out_tensor;

  if (use_probs) {
    TORCH_CHECK(probs.value().scalar_type() == torch::kFloat32,
                "probs must be a FloatTensor");
    TORCH_CHECK(probs.value().is_contiguous(), "probs must be contiguous");

    if (is_cpu) {
      d_probs_tensor = probs.value().to(torch::kCUDA);
    } else {
      d_probs_tensor = probs.value();
    }
    d_probs_out_tensor = torch::empty_like(d_probs_tensor);
    d_probs_in = d_probs_tensor.data_ptr<float>();
    d_probs_out = d_probs_out_tensor.data_ptr<float>();
  }

  int *d_changed;
  cudaMalloc(&d_changed, sizeof(int));
  int *d_marked_count;
  cudaMalloc(&d_marked_count, sizeof(int));

  unsigned int *d_marked_indices = nullptr;
  unsigned char *d_new_values = nullptr;
  unsigned char *h_img = nullptr;

  cudaMalloc(&d_marked_indices, total_size * sizeof(unsigned int));
  if (mode == 1) { // Mode 1: Exact ITK Hybrid
    if (is_cpu) {
      h_img = image.data_ptr<unsigned char>();
    } else {
      h_img = new unsigned char[total_size];
      cudaMemcpy(h_img, d_img, total_size, cudaMemcpyDeviceToHost);
    }
    cudaMalloc(&d_new_values, total_size * sizeof(unsigned char));
  }

  dim3 blockSize(8, 4, 4);
  dim3 gridSize((w + blockSize.x - 1) / blockSize.x,
                (h + blockSize.y - 1) / blockSize.y,
                (d + blockSize.z - 1) / blockSize.z);

  int h_changed = 0;
  do {
    h_changed = 0;
    if (mode == 0) {
      cudaMemset(d_changed, 0, sizeof(int));
    }
    for (int border = 1; border <= 6; ++border) {
      cudaMemset(d_marked_count, 0, sizeof(int));
      mark_deletable_points_kernel<<<gridSize, blockSize>>>(
          d_img, d, h, w, border, d_marked_indices, d_marked_count);

      int h_count = 0;
      cudaMemcpy(&h_count, d_marked_count, sizeof(int), cudaMemcpyDeviceToHost);

      if (h_count > 0) {
        if (mode == 1) {
          // Mode 1: CPU Sequential (Exact ITK Match)
          std::vector<unsigned int> h_marked(h_count);
          cudaMemcpy(h_marked.data(), d_marked_indices,
                     h_count * sizeof(unsigned int), cudaMemcpyDeviceToHost);

          // Sort indices on CPU to maintain ITK's specific processing order
          // (lexicographical)
          std::sort(h_marked.begin(), h_marked.end());

          std::vector<unsigned char> h_new_values(h_count);

          for (int i = 0; i < h_count; ++i) {
            unsigned int idx = h_marked[i];

            int x = idx % w;
            int y = (idx / w) % h;
            int z = idx / (w * h);

            h_img[idx] = 0; // Temporarily delete

            int neighbors[27];
            get_neighbors(h_img, d, h, w, x, y, z, neighbors);

            if (!is_simple_point(neighbors)) {
              h_img[idx] = 1; // Not simple anymore, restore
              h_new_values[i] = 1;
            } else {
              h_new_values[i] = 0;
              h_changed++;
            }
          }

          cudaMemcpy(d_new_values, h_new_values.data(),
                     h_count * sizeof(unsigned char), cudaMemcpyHostToDevice);

          // Also need to copy the sorted indices back for apply_updates_kernel
          cudaMemcpy(d_marked_indices, h_marked.data(),
                     h_count * sizeof(unsigned int), cudaMemcpyHostToDevice);

          int threads = 256;
          int blocks = (h_count + threads - 1) / threads;
          apply_updates_kernel<<<blocks, threads>>>(d_img, d_marked_indices,
                                                    d_new_values, h_count);
        } else if (mode == 0) {
          // Mode 0: GPU Subgrid (8-color parallel) - Topologically safe, purely
          // GPU
          int threads = 256;
          int blocks = (h_count + threads - 1) / threads;
          for (int color = 0; color < 8; ++color) {
            subgrid_recheck_kernel<<<blocks, threads>>>(
                d_img, d, h, w, d_marked_indices, h_count, color, d_changed);
          }
        }
      }
    }
    if (mode == 0) {
      int pass_changed = 0;
      cudaMemcpy(&pass_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
      h_changed += pass_changed;
    }

    if (use_probs && h_changed > 0) {
      inward_sweep_kernel<<<gridSize, blockSize>>>(d_img, d_probs_in,
                                                   d_probs_out, d, h, w);
      std::swap(d_probs_in, d_probs_out);
    }
  } while (h_changed > 0);

  if (is_cpu) {
    image.copy_(d_tensor);
  }

  if (use_probs) {
    if (d_probs_in != d_probs_tensor.data_ptr<float>()) {
      d_probs_tensor.copy_(d_probs_out_tensor);
    }
    if (is_cpu) {
      probs.value().copy_(d_probs_tensor);
    }
  }

  cudaFree(d_marked_indices);
  if (mode == 1) {
    if (!is_cpu) {
      delete[] h_img;
    }
    cudaFree(d_new_values);
  }
  cudaFree(d_changed);
  cudaFree(d_marked_count);
}

__global__ void region_grow_backward_inplace(const unsigned char *mask,
                                             unsigned char *visited,
                                             float *grad, int d, int h, int w,
                                             int *changed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= w || y >= h || z >= d)
    return;

  size_t idx = (size_t)z * (h * w) + y * w + x;

  if (mask[idx] == 0)
    return;
  if (visited[idx] >= 1)
    return; // 1 or 2 means already visited

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;
        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h && nz >= 0 && nz < d) {
          size_t n_idx = (size_t)nz * (h * w) + ny * w + nx;
          if (visited[n_idx] ==
              1) { // Only pull from fully committed visited pixels
            grad[idx] = grad[n_idx];
            visited[idx] = 2; // mark as newly visited for this pass
            atomicAdd(changed, 1);
            return;
          }
        }
      }
    }
  }
}

__global__ void commit_visited_kernel(unsigned char *visited, int d, int h,
                                      int w) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= w || y >= h || z >= d)
    return;
  size_t idx = (size_t)z * (h * w) + y * w + x;
  if (visited[idx] == 2)
    visited[idx] = 1;
}

void region_grow_backward_cuda(torch::Tensor mask, torch::Tensor cl,
                               torch::Tensor grad_S, torch::Tensor grad_prob) {
  TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
  TORCH_CHECK(cl.is_cuda(), "cl must be a CUDA tensor");
  TORCH_CHECK(grad_S.is_cuda(), "grad_S must be a CUDA tensor");
  TORCH_CHECK(grad_prob.is_cuda(), "grad_prob must be a CUDA tensor");

  int d = mask.size(0);
  int h = mask.size(1);
  int w = mask.size(2);

  torch::Tensor visited = cl.clone();

  // grad_prob is pre-allocated by python
  grad_prob.copy_(grad_S);
  grad_prob.mul_(cl); // zero out everything not in skeleton

  int *d_changed;
  cudaMalloc(&d_changed, sizeof(int));

  dim3 blockSize(8, 4, 4);
  dim3 gridSize((w + blockSize.x - 1) / blockSize.x,
                (h + blockSize.y - 1) / blockSize.y,
                (d + blockSize.z - 1) / blockSize.z);

  int h_changed = 0;
  do {
    cudaMemset(d_changed, 0, sizeof(int));
    region_grow_backward_inplace<<<gridSize, blockSize>>>(
        mask.data_ptr<unsigned char>(), visited.data_ptr<unsigned char>(),
        grad_prob.data_ptr<float>(), d, h, w, d_changed);
    cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_changed > 0) {
      commit_visited_kernel<<<gridSize, blockSize>>>(
          visited.data_ptr<unsigned char>(), d, h, w);
    }
  } while (h_changed > 0);

  cudaFree(d_changed);
}

void binary_thinning_cuda(torch::Tensor image, int mode) {
  thinning_internal(image, mode, std::nullopt);
}

void extract_centerline_cuda(torch::Tensor mask, torch::Tensor probs,
                             int mode) {
  thinning_internal(mask, mode, probs);
}
