#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
        size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(T const* A, size_t lda,
                                           T const* B, size_t ldb,
                                           T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                                           T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n, size_t k)
{
    // loading data from A on D_RAM
    #pragma unroll
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS; ++load_idx)
    {
       size_t const A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
       size_t const A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};

       size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_block_tile_row_idx};
       size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};

       // boundary check
       T val{static_cast<T>(0)};
       if (A_row_idx < m && A_col_idx < k)
       {
            val = A[A_row_idx * lda + A_col_idx];
       }
       A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
    }

    // loading data from B on D_ram
    #pragma unroll
    for(size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS; ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};

        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + B_thread_block_tile_col_idx};

        // boundary check
        T val{static_cast<T>(0)};
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K>
__global__ void two_dim_block_tiling(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda, T const* B, size_t ldb, T beta, T* C, size_t ldc)
{
    // get the num threads
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // computet he row and col for C
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};

    // cache tiles
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};

    T sum{static_cast<T>(0)};
    for (size_t thread_block_tile_idx{0U}; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx)
    {
        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile, thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

        #pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            sum += A_thread_block_tile[threadIdx.y][k_i] * B_thread_block_tile[k_i][threadIdx.x];
        }
        __syncthreads();
    }
    // boundary check
    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * ldc + C_col_idx] = alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_two_dim_kernel(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    two_dim_block_tiling<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_two_dim_kernel<float>(size_t m, size_t n, size_t k,
                                            float const* alpha, float const* A,
                                            size_t lda, float const* B,
                                            size_t ldb, float const* beta,
                                            float* C, size_t ldc,
                                            cudaStream_t stream);
template void launch_two_dim_kernel<double>(size_t m, size_t n, size_t k,
                                             double const* alpha,
                                             double const* A, size_t lda,
                                             double const* B, size_t ldb,
                                             double const* beta, double* C,
                                             size_t ldc, cudaStream_t stream);
template void launch_two_dim_kernel<__half>(size_t m, size_t n, size_t k,
                                             __half const* alpha,
                                             __half const* A, size_t lda,
                                             __half const* B, size_t ldb,
                                             __half const* beta, __half* C,
                                             size_t ldc, cudaStream_t stream);