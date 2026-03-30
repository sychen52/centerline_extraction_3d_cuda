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
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
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

__host__ __device__ int uf_find(int i, int *parent) {
  while (parent[i] != i) {
    parent[i] = parent[parent[i]];
    i = parent[i];
  }
  return i;
}

__host__ __device__ void uf_union(int i, int j, int *parent) {
  int root_i = uf_find(i, parent);
  int root_j = uf_find(j, parent);
  if (root_i != root_j) {
    parent[root_i] = root_j;
  }
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

__host__ __device__ bool is_simple_point(const int *neighbors) {
  int parent[27];
  for (int i = 0; i < 27; ++i) {
    parent[i] = i;
  }

  for (int i = 0; i < 27; ++i) {
    if (i == 13 || neighbors[i] != 1)
      continue;
    int x1 = i % 3;
    int y1 = (i / 3) % 3;
    int z1 = i / 9;

    for (int j = i + 1; j < 27; ++j) {
      if (j == 13 || neighbors[j] != 1)
        continue;
      int x2 = j % 3;
      int y2 = (j / 3) % 3;
      int z2 = j / 9;

      if (abs(x1 - x2) <= 1 && abs(y1 - y2) <= 1 && abs(z1 - z2) <= 1) {
        uf_union(i, j, parent);
      }
    }
  }

  int components = 0;
  for (int i = 0; i < 27; ++i) {
    if (i == 13)
      continue;
    if (neighbors[i] == 1 && parent[i] == i) {
      components++;
    }
  }

  return (components <= 1);
}

__global__ void mark_deletable_points_kernel(unsigned char *img, int d, int h,
                                             int w, int currentBorder) {
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
}

struct is_marked {
  unsigned char *img;
  __host__ __device__ is_marked(unsigned char *_img) : img(_img) {}
  __device__ bool operator()(const size_t &idx) const { return img[idx] == 2; }
};

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

void binary_thinning_cuda(torch::Tensor image, int mode) {
  TORCH_CHECK(image.is_cuda(), "image must be a CUDA tensor");
  TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
  TORCH_CHECK(image.scalar_type() == torch::kByte,
              "image must be a ByteTensor (uint8)");
  TORCH_CHECK(image.dim() == 3, "image must be a 3D tensor");

  int d = image.size(0);
  int h = image.size(1);
  int w = image.size(2);
  size_t total_size = (size_t)d * h * w;

  unsigned char *d_img = image.data_ptr<unsigned char>();

  int *d_changed;
  cudaMalloc(&d_changed, sizeof(int));

  unsigned int *d_marked_indices = nullptr;
  unsigned char *d_new_values = nullptr;
  unsigned char *h_img = nullptr;

  cudaMalloc(&d_marked_indices, total_size * sizeof(unsigned int));
  if (mode == 1) { // Mode 1: Exact ITK Hybrid
    h_img = new unsigned char[total_size];
    cudaMemcpy(h_img, d_img, total_size, cudaMemcpyDeviceToHost);
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
      mark_deletable_points_kernel<<<gridSize, blockSize>>>(d_img, d, h, w,
                                                            border);

      thrust::counting_iterator<size_t> first(0);
      thrust::counting_iterator<size_t> last(total_size);
      thrust::device_ptr<unsigned int> dest(d_marked_indices);

      auto end_ptr =
          thrust::copy_if(thrust::device, first, last, dest, is_marked(d_img));
      int h_count = end_ptr - dest;

      if (h_count > 0) {
        if (mode == 1) {
          // Mode 1: CPU Sequential (Exact ITK Match)
          std::vector<unsigned int> h_marked(h_count);
          cudaMemcpy(h_marked.data(), d_marked_indices,
                     h_count * sizeof(unsigned int), cudaMemcpyDeviceToHost);

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
  } while (h_changed > 0);

  cudaFree(d_marked_indices);
  if (mode == 1) {
    delete[] h_img;
    cudaFree(d_new_values);
  }
  cudaFree(d_changed);
}
