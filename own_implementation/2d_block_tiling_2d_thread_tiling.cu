#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_X, size_t THREAD_TILE_SIZE_Y>
__global__ void two_dim_tiling_two_dim_thread_kernel(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda, T const* B, size_t ldb, T beta, T* C, size_t ldc)
{
    // num of threads and thread linear idx
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // cache tiles
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};

    T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    // A vals and B vals are cached in registers
    T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    T B_vals[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U}; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx)
    {
        load_data_from_global_memory_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile, thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

        #pragma unroll
        for(size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            size_t A_thread_block_tile_row_idx{thread_tile_row_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y};
            size_t const A_thread_block_tile_col_idx{k_i};

            #pragma unroll
            for(size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
            {
                A_vals[thread_tile_row_idx] = A_thread_block_tile[A_thread_block_tile_row_idx + thread_tile_row_idx][A_thread_block_tile_col_idx + thread_tile_col_idx];
            }
            
            size_t const B_thread_block_tile_row_idx{k_i};
            size_t const B_thread_block_tile_col_idx{thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X};

            #pragma unroll
            for(size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx)
            {
                B_vals[thread_tile_col_idx] = B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx + thread_tile_col_idx];
            }

            for(size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
            {
                for(size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx)
                {
                    C_thread_results[thread_tile_row_idx][thread_tile_col_idx] += A_vals[thread_tile_row_idx] * B_vals[thread_tile_col_idx];
                }
            }
        }
        __syncthreads();
    }

    // write the results to DRAM
    for(size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
    {
        for(size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx)
        {
            size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y + thread_tile_row_idx};
            size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + threadIdx.y / (BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y) * THREAD_TILE_SIZE_X + thread_tile_col_idx};

            if (C_row_idx < m && C_col_idx < n)
            {
                C[C_row_idx * ldc + C_col_idx] alpha * C_thread_results[thread_tile_row_idx][thread_tile_col_idx] + beta * C[C_row_idx * ldc + C_col_idx];
            }
        }
    }
}

template <typename T>
void launch_two_dim_tiling_two_dim_thread_kernel(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
    // Each thread computes THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y values of C.
    constexpr unsigned int THREAD_TILE_SIZE_X{8U};
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
        (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(
        BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    two_dim_tiling_two_dim_thread_kernel<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
             THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_two_dim_tiling_two_dim_thread_kernel<float>(size_t m, size_t n, size_t k,
                                            float const* alpha, float const* A,
                                            size_t lda, float const* B,
                                            size_t ldb, float const* beta,
                                            float* C, size_t ldc,
                                            cudaStream_t stream);
template void launch_two_dim_tiling_two_dim_thread_kernel<double>(size_t m, size_t n, size_t k,
                                             double const* alpha,
                                             double const* A, size_t lda,
                                             double const* B, size_t ldb,
                                             double const* beta, double* C,
                                             size_t ldc, cudaStream_t stream);
template void launch_two_dim_tiling_two_dim_thread_kernel<__half>(size_t m, size_t n, size_t k,
                                             __half const* alpha,
                                             __half const* A, size_t lda,
                                             __half const* B, size_t ldb,
                                             __half const* beta, __half* C,
                                             size_t ldc, cudaStream_t stream);