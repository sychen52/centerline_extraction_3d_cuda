#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int d_eulerLUT[256];

void fill_euler_lut(int* LUT) {
    for(int i=0; i<256; i++) LUT[i] = 0;
    LUT[1] = 1; LUT[3] = -1; LUT[5] = -1; LUT[7] = 1; LUT[9] = -3; LUT[11] = -1; LUT[13] = -1; LUT[15] = 1;
    LUT[17] = -1; LUT[19] = 1; LUT[21] = 1; LUT[23] = -1; LUT[25] = 3; LUT[27] = 1; LUT[29] = 1; LUT[31] = -1;
    LUT[33] = -3; LUT[35] = -1; LUT[37] = 3; LUT[39] = 1; LUT[41] = 1; LUT[43] = -1; LUT[45] = 3; LUT[47] = 1;
    LUT[49] = -1; LUT[51] = 1; LUT[53] = 1; LUT[55] = -1; LUT[57] = 3; LUT[59] = 1; LUT[61] = 1; LUT[63] = -1;
    LUT[65] = -3; LUT[67] = 3; LUT[69] = -1; LUT[71] = 1; LUT[73] = 1; LUT[75] = 3; LUT[77] = -1; LUT[79] = 1;
    LUT[81] = -1; LUT[83] = 1; LUT[85] = 1; LUT[87] = -1; LUT[89] = 3; LUT[91] = 1; LUT[93] = 1; LUT[95] = -1;
    LUT[97] = 1; LUT[99] = 3; LUT[101] = 3; LUT[103] = 1; LUT[105] = 5; LUT[107] = 3; LUT[109] = 3; LUT[111] = 1;
    LUT[113] = -1; LUT[115] = 1; LUT[117] = 1; LUT[119] = -1; LUT[121] = 3; LUT[123] = 1; LUT[125] = 1; LUT[127] = -1;
    LUT[129] = -7; LUT[131] = -1; LUT[133] = -1; LUT[135] = 1; LUT[137] = -3; LUT[139] = -1; LUT[141] = -1; LUT[143] = 1;
    LUT[145] = -1; LUT[147] = 1; LUT[149] = 1; LUT[151] = -1; LUT[153] = 3; LUT[155] = 1; LUT[157] = 1; LUT[159] = -1;
    LUT[161] = -3; LUT[163] = -1; LUT[165] = 3; LUT[167] = 1; LUT[169] = 1; LUT[171] = -1; LUT[173] = 3; LUT[175] = 1;
    LUT[177] = -1; LUT[179] = 1; LUT[181] = 1; LUT[183] = -1; LUT[185] = 3; LUT[187] = 1; LUT[189] = 1; LUT[191] = -1;
    LUT[193] = -3; LUT[195] = 3; LUT[197] = -1; LUT[199] = 1; LUT[201] = 1; LUT[203] = 3; LUT[205] = -1; LUT[207] = 1;
    LUT[209] = -1; LUT[211] = 1; LUT[213] = 1; LUT[215] = -1; LUT[217] = 3; LUT[219] = 1; LUT[221] = 1; LUT[223] = -1;
    LUT[225] = 1; LUT[227] = 3; LUT[229] = 3; LUT[231] = 1; LUT[233] = 5; LUT[235] = 3; LUT[237] = 3; LUT[239] = 1;
    LUT[241] = -1; LUT[243] = 1; LUT[245] = 1; LUT[247] = -1; LUT[249] = 3; LUT[251] = 1; LUT[253] = 1; LUT[255] = -1;
}

__device__ bool is_euler_invariant(const int* neighbors) {
    int eulerChar = 0;
    unsigned char n;

    // Octant SWU
    n = 1;
    if (neighbors[24] == 1) n |= 128;
    if (neighbors[25] == 1) n |= 64;
    if (neighbors[15] == 1) n |= 32;
    if (neighbors[16] == 1) n |= 16;
    if (neighbors[21] == 1) n |= 8;
    if (neighbors[22] == 1) n |= 4;
    if (neighbors[12] == 1) n |= 2;
    eulerChar += d_eulerLUT[n];

    // Octant SEU
    n = 1;
    if (neighbors[26] == 1) n |= 128;
    if (neighbors[23] == 1) n |= 64;
    if (neighbors[17] == 1) n |= 32;
    if (neighbors[14] == 1) n |= 16;
    if (neighbors[25] == 1) n |= 8;
    if (neighbors[22] == 1) n |= 4;
    if (neighbors[16] == 1) n |= 2;
    eulerChar += d_eulerLUT[n];

    // Octant NWU
    n = 1;
    if (neighbors[18] == 1) n |= 128;
    if (neighbors[21] == 1) n |= 64;
    if (neighbors[9] == 1) n |= 32;
    if (neighbors[12] == 1) n |= 16;
    if (neighbors[19] == 1) n |= 8;
    if (neighbors[22] == 1) n |= 4;
    if (neighbors[10] == 1) n |= 2;
    eulerChar += d_eulerLUT[n];

    // Octant NEU
    n = 1;
    if (neighbors[20] == 1) n |= 128;
    if (neighbors[23] == 1) n |= 64;
    if (neighbors[19] == 1) n |= 32;
    if (neighbors[22] == 1) n |= 16;
    if (neighbors[11] == 1) n |= 8;
    if (neighbors[14] == 1) n |= 4;
    if (neighbors[10] == 1) n |= 2;
    eulerChar += d_eulerLUT[n];

    // Octant SWB
    n = 1;
    if (neighbors[6] == 1) n |= 128;
    if (neighbors[15] == 1) n |= 64;
    if (neighbors[7] == 1) n |= 32;
    if (neighbors[16] == 1) n |= 16;
    if (neighbors[3] == 1) n |= 8;
    if (neighbors[12] == 1) n |= 4;
    if (neighbors[4] == 1) n |= 2;
    eulerChar += d_eulerLUT[n];

    // Octant SEB
    n = 1;
    if (neighbors[8] == 1) n |= 128;
    if (neighbors[7] == 1) n |= 64;
    if (neighbors[17] == 1) n |= 32;
    if (neighbors[16] == 1) n |= 16;
    if (neighbors[5] == 1) n |= 8;
    if (neighbors[4] == 1) n |= 4;
    if (neighbors[14] == 1) n |= 2;
    eulerChar += d_eulerLUT[n];

    // Octant NWB
    n = 1;
    if (neighbors[0] == 1) n |= 128;
    if (neighbors[9] == 1) n |= 64;
    if (neighbors[3] == 1) n |= 32;
    if (neighbors[12] == 1) n |= 16;
    if (neighbors[1] == 1) n |= 8;
    if (neighbors[10] == 1) n |= 4;
    if (neighbors[4] == 1) n |= 2;
    eulerChar += d_eulerLUT[n];

    // Octant NEB
    n = 1;
    if (neighbors[2] == 1) n |= 128;
    if (neighbors[1] == 1) n |= 64;
    if (neighbors[11] == 1) n |= 32;
    if (neighbors[10] == 1) n |= 16;
    if (neighbors[5] == 1) n |= 8;
    if (neighbors[4] == 1) n |= 4;
    if (neighbors[14] == 1) n |= 2;
    eulerChar += d_eulerLUT[n];

    return (eulerChar == 0);
}

__device__ int uf_find(int i, int* parent) {
    while(parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

__device__ void uf_union(int i, int j, int* parent) {
    int root_i = uf_find(i, parent);
    int root_j = uf_find(j, parent);
    if(root_i != root_j) {
        parent[root_i] = root_j;
    }
}

__device__ bool is_simple_point(const int* neighbors) {
    int parent[27];
    for (int i = 0; i < 27; ++i) {
        parent[i] = i;
    }

    for (int i = 0; i < 27; ++i) {
        if (i == 13 || neighbors[i] != 1) continue;
        int x1 = i % 3;
        int y1 = (i / 3) % 3;
        int z1 = i / 9;

        for (int j = i + 1; j < 27; ++j) {
            if (j == 13 || neighbors[j] != 1) continue;
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
        if (i == 13) continue;
        if (neighbors[i] == 1 && parent[i] == i) {
            components++;
        }
    }

    return (components <= 1);
}

__global__ void mark_deletable_points_kernel(
    unsigned char* img, 
    int d, int h, int w, 
    int currentBorder) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= w || y >= h || z >= d) return;

    size_t idx = z * (h * w) + y * w + x;
    if (img[idx] != 1) return;

    int neighbors[27];
    int num_neighbors = -1;
    bool isBorderPoint = false;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                
                int n_idx = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
                
                int val = 0;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h && nz >= 0 && nz < d) {
                    size_t flat_n_idx = nz * (h * w) + ny * w + nx;
                    val = img[flat_n_idx];
                }
                
                int binary_val = (val == 1) ? 1 : 0;
                neighbors[n_idx] = binary_val;
                
                if (binary_val == 1) {
                    num_neighbors++;
                }

                if (dx == 0 && dy == -1 && dz == 0 && currentBorder == 1 && val <= 0) isBorderPoint = true; // N
                if (dx == 0 && dy == 1 && dz == 0 && currentBorder == 2 && val <= 0) isBorderPoint = true;  // S
                if (dx == 1 && dy == 0 && dz == 0 && currentBorder == 3 && val <= 0) isBorderPoint = true;  // E
                if (dx == -1 && dy == 0 && dz == 0 && currentBorder == 4 && val <= 0) isBorderPoint = true; // W
                if (dx == 0 && dy == 0 && dz == 1 && currentBorder == 5 && val <= 0) isBorderPoint = true;  // U
                if (dx == 0 && dy == 0 && dz == -1 && currentBorder == 6 && val <= 0) isBorderPoint = true; // B
            }
        }
    }

    if (!isBorderPoint) return;
    if (num_neighbors == 1) return;
    if (!is_euler_invariant(neighbors)) return;
    if (!is_simple_point(neighbors)) return;

    img[idx] = 2; // mark for deletion
}

__global__ void remove_marked_points_kernel(unsigned char* img, int d, int h, int w, int* changed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= w || y >= h || z >= d) return;

    size_t idx = z * (h * w) + y * w + x;
    if (img[idx] == 2) {
        img[idx] = 0;
        atomicAdd(changed, 1);
    }
}

void binary_thinning_cuda(torch::Tensor image) {
    TORCH_CHECK(image.is_cuda(), "image must be a CUDA tensor");
    TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
    TORCH_CHECK(image.scalar_type() == torch::kByte, "image must be a ByteTensor (uint8)");
    TORCH_CHECK(image.dim() == 3, "image must be a 3D tensor");

    int d = image.size(0);
    int h = image.size(1);
    int w = image.size(2);

    unsigned char* d_img = image.data_ptr<unsigned char>();

    static bool lut_initialized = false;
    if (!lut_initialized) {
        int eulerLUT_host[256];
        fill_euler_lut(eulerLUT_host);
        cudaMemcpyToSymbol(d_eulerLUT, eulerLUT_host, 256 * sizeof(int));
        lut_initialized = true;
    }

    int* d_changed;
    cudaMalloc(&d_changed, sizeof(int));

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x,
                  (h + blockSize.y - 1) / blockSize.y,
                  (d + blockSize.z - 1) / blockSize.z);

    int h_changed = 0;
    do {
        h_changed = 0;
        for (int border = 1; border <= 6; ++border) {
            mark_deletable_points_kernel<<<gridSize, blockSize>>>(d_img, d, h, w, border);
            
            cudaMemset(d_changed, 0, sizeof(int));
            remove_marked_points_kernel<<<gridSize, blockSize>>>(d_img, d, h, w, d_changed);
            
            int changed_this_border = 0;
            cudaMemcpy(&changed_this_border, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
            h_changed += changed_this_border;
        }
    } while (h_changed > 0);

    cudaFree(d_changed);
}
