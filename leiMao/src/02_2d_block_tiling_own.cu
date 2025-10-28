#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(T const* A, size_t lda,
                                           T const* B, size_t ldb,
                                           T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                                           T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n,
                                           size_t k)
{
    // TODO: Load a tile of matrix A from global memory to shared memory A_thread_block_tile.
    // Each thread should cooperatively load BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K elements.
    // Use thread_linear_idx and NUM_THREADS to distribute work across the thread block.
    // Calculate the row and column indices in both A_thread_block_tile and the original matrix A.
    // Perform boundary checks: if A_row_idx < m && A_col_idx < k, load the value; otherwise, load 0.
    // The loop should iterate (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1) / NUM_THREADS times.

    // YOUR CODE HERE
#pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS; ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K
            };
            size_t const A_thread_block_tile_col_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K
            };
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};
            
            // boundary checks
            if (A_row_idx < m && A_col_idx < k)
            {
                T val = A[A_row_idx * lda + A_col_idx];
            }
            A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
        }


    // TODO: Load a tile of matrix B from global memory to shared memory B_thread_block_tile.
    // Each thread should cooperatively load BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X elements.
    // Use thread_linear_idx and NUM_THREADS to distribute work across the thread block.
    // Calculate the row and column indices in both B_thread_block_tile and the original matrix B.
    // Perform boundary checks: if B_row_idx < k && B_col_idx < n, load the value; otherwise, load 0.
    // The loop should iterate (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1) / NUM_THREADS times.

    // YOUR CODE HERE
#pragma unroll
        for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS; ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X
            };
            size_t const B_thread_block_tile_col_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X
            };
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + B_thread_block_tile_col_idx};

            // boundary checks
            if (B_row_idx < k && B_col_idx < n)
            {
                T val = B[B_row_idx * ldb + B_col_idx];
            }
            B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
        }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // TODO: Define NUM_THREADS as a compile-time constant equal to BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y.
    // This allows the compiler to optimize loop unrolling better than using blockDim.x * blockDim.y.

    // YOUR CODE HERE
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};


    // TODO: Calculate the linear thread index from blockIdx, blockDim, threadIdx.
    // This is used for cooperative loading of data to shared memory.

    // YOUR CODE HERE
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};


    // TODO: Compute the column index (C_col_idx) and row index (C_row_idx) of matrix C
    // that this thread is responsible for. Use blockIdx and threadIdx.

    // YOUR CODE HERE
    size_t const C_col_idx{blockIdx.x * blockDim.y + threadIdx.y};
    size_t const C_row_idx{blockIdx.y * blockDim.x + threadIdx.x};


    // TODO: Declare shared memory tiles for A and B.
    // A_thread_block_tile should have dimensions BLOCK_TILE_SIZE_Y x BLOCK_TILE_SIZE_K.
    // B_thread_block_tile should have dimensions BLOCK_TILE_SIZE_K x BLOCK_TILE_SIZE_X.

    // YOUR CODE HERE
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];


    // TODO: Calculate the number of thread block tiles needed to cover dimension k.
    // This is: (k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K

    // YOUR CODE HERE
    size_t const num_thread_block_tiles{
        (k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K
    };


    // TODO: Initialize the accumulator sum to zero.

    // YOUR CODE HERE
    T sum{static_cast<T>(0)};


    // TODO: Iterate over each thread block tile of k dimension.
    // For each iteration:
    //   1. Call load_data_to_shared_memory to load tiles of A and B into shared memory.
    //   2. Call __syncthreads() to ensure all threads have finished loading.
    //   3. Compute the partial dot product by iterating over BLOCK_TILE_SIZE_K
    //      and accumulating: sum += A_thread_block_tile[threadIdx.y][k_i] * B_thread_block_tile[k_i][threadIdx.x]
    //   4. Call __syncthreads() before moving to the next tile.

    // YOUR CODE HERE
    for (size_t thread_block_tile_idx{0U}; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx)
    {
        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile, thread_block_tile_idx, thread_linear_idx, m, n, k
        );
        __syncthreads();
    }
#pragma unroll
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    {
        sum += A_thread_block_tile[threadIdx.y][k_i] * B_thread_block_tile[k_i][threadIdx.x];
    }
    __syncthreads();

    // TODO: Store the result back to matrix C.
    // Perform boundary check: if C_row_idx < m && C_col_idx < n
    // Compute: C[C_row_idx * ldc + C_col_idx] = alpha * sum + beta * C[C_row_idx * ldc + C_col_idx]

    // YOUR CODE HERE
    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * ldc + C_col_idx] = alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // TODO: Define compile-time constants for tile sizes.
    // Set BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, and BLOCK_TILE_SIZE_K.
    // These control the size of the tiles loaded into shared memory.
    // Typical values are 32x32x32 for good hardware utilization.

    // YOUR CODE HERE
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};


    // TODO: Define NUM_THREADS as BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y.

    // YOUR CODE HERE
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};

    // TODO: Add static assertions to verify that the tile dimensions are compatible with NUM_THREADS.
    // Both of these should hold:
    //   - BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y must be divisible by NUM_THREADS
    //   - BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K must be divisible by NUM_THREADS

    // YOUR CODE HERE
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);

    // TODO: Create the block_dim (thread block dimensions) using dim3.
    // Set x to BLOCK_TILE_SIZE_X, y to BLOCK_TILE_SIZE_Y, and z to 1.

    // YOUR CODE HERE
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};

    // TODO: Create the grid_dim (grid dimensions) using dim3.
    // Grid x dimension: (n + block_dim.x - 1) / block_dim.x
    // Grid y dimension: (m + block_dim.y - 1) / block_dim.y
    // Grid z dimension: 1
    // Cast m and n to unsigned int to avoid issues.

    // YOUR CODE HERE
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U
    };

    // TODO: Launch the gemm_v02 kernel with the computed grid and block dimensions.
    // Pass the kernel function, grid_dim, block_dim, shared memory size (0),
    // stream, and all the parameters (m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc).
    // Call CHECK_LAST_CUDA_ERROR() after launching.

    // YOUR CODE HERE
    gemm_v02<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
