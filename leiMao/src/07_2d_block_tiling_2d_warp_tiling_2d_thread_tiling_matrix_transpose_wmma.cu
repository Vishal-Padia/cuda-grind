#include <cuda_fp16.h>
#include <mma.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t BLOCK_TILE_SKEW_SIZE_X, size_t BLOCK_TILE_SKEW_SIZE_Y, size_t WARP_TILE_SIZE_X, size_t WARP_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_X, size_t WMMA_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_K, size_t NUM_THREADS>
__global__ void gemm_v07_vectorized(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda, T const* B, size_t ldb, T beta, T* C, size_t ldc)
{
    // static assertions
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    // cache a tile of A and B in shared memory
    __shared__ T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X];

    constexpr size_t NUM_WMMA_TILES_X{WARP_TILE_SIZE_X / WMMA_TILE_SIZE_X};
    static_assert(WARP_TILE_SIZE_X % WMMA_TILE_SIZE_X == 0U);
    constexpr size_t NUM_WMMA_TILES_Y{WARP_TILE_SIZE_Y / WMMA_TILE_SIZE_Y};
    static_assert(WARP_TILE_SIZE_Y % WMMA_TILE_SIZE_Y == 0U);
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K};
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0U);

    // declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, // use: A matrix from A*B
    WMMA_TILE_SIZE_Y, // M dimension
    WMMA_TILE_SIZE_X, // N dimension
    WMMA_TILE_SIZE_K, // K dimension
    T, // element type
    nvcuda::wmma::col_major>a_frags[NUM_WMMA_TILES_Y];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T, nvcuda::wmma::row_major>b_frags[NUM_WMMA_TILES_X]; // intialize a fragment for the B matrix
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K, T>acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X]; // intialize a fragment for the accumulator
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>c_frag; // intialize a fragment for the output

    // make sure the accumulator starts from 0
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
    {
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
        {
            // sets all elements in the fragment to a constant value (0 in this case)
            nvcuda::wmma::fill_fragment(acc_frags[wmma_tile_row_idx][wmma_tile_col_idx], static_cast<T>(0));
        }
    }
    
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x}; // linear index of the thread in the block
    size_t const warp_linear_idx{thread_linear_idx / 32U}; // linear index of the warp in the block
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X}; // row index of the warp in the block
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X}; // column index of the warp in the block

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K}; // number of thread block tiles needed to cover the entire k dimension

    for (size_t thread_block_tile_idx{0U}; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx)
    {
        load_data_from_global_memory_to_shared_memory_transposed_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS, BLOCK_TILE_SKEW_SIZE_X, BLOCK_TILE_SKEW_SIZE_Y>(A, lda, B, ldb, A_thread_block_tile_transposed, B_thread_block_tile,thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < NUM_WMMA_TILES_K; ++k_i)
        {
#pragma unroll
            for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
            {
                // loads a matrix tile from memory into a fragment
                // we provide a pointer to the logical start to a matrix tile 
                // and the WMMA API figures out which thread loads which elements
                // to satisfy the interal distribution pattern
                nvcuda::wmma::load_matrix_sync(a_frags[wmma_tile_row_idx], &A_thread_block_tile_transposed[k_i * WMMA_TILE_SIZE_K][warp_row_idx * WARP_TILE_SIZE_Y + wmma_tile_row_idx * WMMA_TILE_SIZE_Y], BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y);
#pragma unroll
                for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
                {
                    // these loads are extremely slow somehow, which affects the performance a lot. Load the fragment from shared memory
                    nvcuda::wmma::load_matrix_sync(b_frags[wmma_tile_col_idx], &B_thread_block_tile[k_i * WMMA_TILE_SIZE_K][warp_col_idx * WARP_TILE_SIZE_X + wmma_tile_col_idx * WMMA_TILE_SIZE_Y], BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X);

                    // perform the matrix multiplication
                    // one mma_sync calls performs a 16x16x16 matrix multiply in a single instruction
                    nvcuda::wmma::mma_sync(
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx], // output fragment
                        a_frags[wmma_tile_row_idx], // Input A fragment
                        b_frags[wmma_tile_col_idx], // Input B fragment
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx]); // previous C value
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    // write results to DRAM
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
    {
#pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
        {
            // load the fragment from shared memory
            nvcuda::wmma::load_matrix_sync(
                c_frag,
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                n, nvcuda::wmma::mem_row_major);

            // perform scaling and addition
            for (size_t i{0}; i < c_frag.num_elements; ++i)
            {
                c_frag.x[i] = alpha * acc_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] + beta * c_frag.x[i];
            }

            // store the fragment back to shared memory
            nvcuda::wmma::store_matrix_sync(
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                c_frag, n, nvcuda::wmma::mem_row_major);
        }
    }
}

template <typename T>
void launch_gemm_kernel_v07_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    // The skew size is used to avoid bank conflicts in shared memory.
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{16U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_Y{16U};

    constexpr unsigned int WARP_TILE_SIZE_X{32U};
    constexpr unsigned int WARP_TILE_SIZE_Y{64U};
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    constexpr unsigned int WMMA_TILE_SIZE_X{16U};
    constexpr unsigned int WMMA_TILE_SIZE_Y{16U};
    constexpr unsigned int WMMA_TILE_SIZE_K{16U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y *
                                                 32U};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v07_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K, BLOCK_TILE_SKEW_SIZE_X,
                        BLOCK_TILE_SKEW_SIZE_Y, WARP_TILE_SIZE_X,
                        WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_Y,
                        WMMA_TILE_SIZE_K, NUM_THREADS_PER_BLOCK>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v07_vectorized<__half>(
    size_t m, size_t n, size_t k, __half const* alpha, __half const* A,
    size_t lda, __half const* B, size_t ldb, __half const* beta, __half* C,
    size_t ldc, cudaStream_t stream);