#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

template <typename T>
__global__ void non_coalesced_memory(size_t m, size_t n, size_t k, T alpha, T const* A,
    size_t lda, T const* B, size_t ldb, T beta, T* C,
    size_t ldc)
{
    // each thread computes one element each
    size_t const row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // boundary check
    if (row_idx < m && col_idx < n)
    {
        // set the sum to 0
        T sum{static_cast<T>(0)};

        // compute the partial sum
        for (size_t k_idx{0U}; k_idx < k; ++k_idx){
            sum += A[row_idx * lda + k_idx] * B[k_idx * ldb + col_idx];
        }
        C[row_idx * ldc + col_idx] = alpha * sum + beta * C[row_idx * ldc + col_idx];
    }
}

template <typename T>
void launch_non_coalesced_memory(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    CHECK_LAST_CUDA_ERROR();
}


// Explicit instantiation.
template void launch_non_coalesced_memory<float>(size_t m, size_t n, size_t k,
                                            float const* alpha, float const* A,
                                            size_t lda, float const* B,
                                            size_t ldb, float const* beta,
                                            float* C, size_t ldc,
                                            cudaStream_t stream);
template void launch_non_coalesced_memory<double>(size_t m, size_t n, size_t k,
                                             double const* alpha,
                                             double const* A, size_t lda,
                                             double const* B, size_t ldb,
                                             double const* beta, double* C,
                                             size_t ldc, cudaStream_t stream);
template void launch_non_coalesced_memory<__half>(size_t m, size_t n, size_t k,
                                             __half const* alpha,
                                             __half const* A, size_t lda,
                                             __half const* B, size_t ldb,
                                             __half const* beta, __half* C,
                                             size_t ldc, cudaStream_t stream);