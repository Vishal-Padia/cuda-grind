#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"


template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(T const* A, size_t lda, T const* B, size_t ldb,
    T A_thread_blocktile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    T B_thread_blocktile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx,
    size_t thread_linear_idx,
    size_t m, size_t n, size_t k
)
{
    // load data from A on DRAM to A_thread_block_tile on shared memory
    #pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS;
        ++load_idx
        )
        {
            size_t const A_thread_blocktile_row_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K
            };
            size_t const A_thread_blocktile_col_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K
            };
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_blocktile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_blocktile_col_idx};

            // boundary checks
            T val{static_cast<T>(0)};
            if (A_row_idx < m && A_col_idx < k)
            {
                val = A[A_row_idx * lda + A_col_idx];
            }
            // adding static assets from the host code to guarantee this if is always true
            static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
            A_thread_blocktile[A_thread_blocktile_row_idx][A_thread_blocktile_col_idx] = val;
        }

        // load data from B on DRAM to B_thread_block_tile on shared memory
        #pragma unroll
        for (size_t load_idx{0U};
             load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS;
            ++load_idx
            )
            {
                size_t const B_thread_blocktile_row_idx{
                    (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X
                };
                size_t const B_thread_blocktile_col_idx{
                    (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X
                };
                size_t const B_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + B_thread_blocktile_row_idx};
                size_t const B_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_blocktile_col_idx};
    
                // boundary checks
                T val{static_cast<T>(0)};
                if (B_row_idx < m && B_col_idx < k)
                {
                    val = B[B_row_idx * ldb + B_col_idx];
                }
                // adding static assets from the host code to guarantee this if is always true
                static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
                B_thread_blocktile[B_thread_blocktile_row_idx][B_thread_blocktile_col_idx] = val;
            } 
}

// coalesced read-write from global memory
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(size_t m, size_t n ,size_t k, T alpha, T const* A, size_t lda, T const* B, size_t ldb, T beta, T* C, size_t ldc)
{
    // avoid using block var as the number of threads per block
    // because it is a runtime constand and the compiler cannot optimize the
    // loop unrolling based on that
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_idx{threadIdx.x * blockDim.x + threadIdx.x};

    // compute the row and col of C
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // cache a tile of A and B in shared memory for data reuse
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1)/ BLOCK_TILE_SIZE_K};

    T sum{static_cast<T>(0)};
    for (size_t thread_block_tile_idx{0U};
        thread_block_tile_idx < num_thread_block_tiles;
        ++thread_block_tile_idx
    )
    {
        load_data_to_shared_memory<T BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile, thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

        #pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            sum += A_thread_block_tile[thread_Idx.y][k_i] * B_thread_block_tile[k_i][thread_Idx.x];
        }
        __syncthreads();
    }
    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * ldc + C_col_idx] = alpha * sum + beta * C[c_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k, T const* alpha,
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
    gemm_v02<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
    <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
