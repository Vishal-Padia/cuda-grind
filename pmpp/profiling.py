import torch
from torch.utils.cpp_extension import load_inline
import time

# Define the CUDA kernel code as strings
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Tiled matrix multiplication kernel with shared memory
__global__ void MatrixMulKernelTiled(float* d_M, float* d_N, float* d_P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    
    // Loop over tiles
    for (int m = 0; m < Width/TILE_WIDTH; ++m) {
        Mds[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[Row * Width + Col] = Pvalue;
}

// Basic matrix multiplication kernel without tiling
__global__ void MatrixMultiplication(float* A, float* B, float* C, int WIDTH) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((row < WIDTH) && (col < WIDTH)) {
        float sum = 0.0f;
        for (int k = 0; k < WIDTH; ++k) {
            sum += A[row * WIDTH + k] * B[k * WIDTH + col];
        }
        C[row * WIDTH + col] = sum;
    }
}

// Wrapper functions to call from Python
torch::Tensor matmul_tiled(torch::Tensor M, torch::Tensor N) {
    const int width = M.size(0);
    auto P = torch::zeros({width, width}, M.options());
    
    // Configure kernel launch parameters
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);
    
    MatrixMulKernelTiled<<<dimGrid, dimBlock>>>(
        M.data_ptr<float>(),
        N.data_ptr<float>(),
        P.data_ptr<float>(),
        width
    );
    
    return P;
}

torch::Tensor matmul_basic(torch::Tensor A, torch::Tensor B) {
    const int width = A.size(0);
    auto C = torch::zeros({width, width}, A.options());
    
    // Configure kernel launch parameters
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + 15) / 16, (width + 15) / 16);
    
    MatrixMultiplication<<<dimGrid, dimBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        width
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_tiled", &matmul_tiled, "Tiled matrix multiplication (CUDA)");
    m.def("matmul_basic", &matmul_basic, "Basic matrix multiplication (CUDA)");
}
"""

# Load and compile the CUDA kernels inline
print("Compiling CUDA kernels... (this may take a minute)")
cuda_module = load_inline(
    name='matrix_mul_kernels',
    cpp_sources='',
    cuda_sources=cuda_source,
    functions=['matmul_tiled', 'matmul_basic'],
    verbose=True,
    extra_cuda_cflags=['-O3']
)

def benchmark_function(func, *args, num_warmup=10, num_iterations=100):
    """
    Benchmark a function with proper GPU synchronization.
    Warmup runs help ensure the GPU is at full performance state.
    """
    # Warmup phase to get GPU into consistent state
    for _ in range(num_warmup):
        result = func(*args)
        torch.cuda.synchronize()
    
    # Actual timing phase
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        result = func(*args)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time, result

def profile_matrix_multiplication(size=1024):
    """
    Profile three matrix multiplication approaches:
    1. Basic CUDA kernel (no shared memory)
    2. Tiled CUDA kernel (with shared memory optimization)
    3. PyTorch's built-in matmul
    """
    print(f"\n{'='*70}")
    print(f"Profiling Matrix Multiplication - Size: {size}x{size}")
    print(f"{'='*70}\n")
    
    # Create random matrices on GPU
    A = torch.randn(size, size, device='cuda', dtype=torch.float32)
    B = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    # Profile basic CUDA kernel
    print("1. Basic CUDA Kernel (no shared memory)...")
    time_basic, result_basic = benchmark_function(cuda_module.matmul_basic, A, B)
    print(f"   Average time: {time_basic*1000:.4f} ms")
    
    # Profile tiled CUDA kernel
    print("\n2. Tiled CUDA Kernel (with shared memory)...")
    time_tiled, result_tiled = benchmark_function(cuda_module.matmul_tiled, A, B)
    print(f"   Average time: {time_tiled*1000:.4f} ms")
    
    # Profile PyTorch matmul
    print("\n3. PyTorch matmul...")
    time_pytorch, result_pytorch = benchmark_function(torch.matmul, A, B)
    print(f"   Average time: {time_pytorch*1000:.4f} ms")
    
    # Calculate speedups
    print(f"\n{'='*70}")
    print("Performance Comparison:")
    print(f"{'='*70}")
    speedup_tiled_vs_basic = time_basic / time_tiled
    speedup_pytorch_vs_basic = time_basic / time_pytorch
    speedup_pytorch_vs_tiled = time_tiled / time_pytorch
    
    print(f"Tiled vs Basic:   {speedup_tiled_vs_basic:.2f}x speedup")
    print(f"PyTorch vs Basic: {speedup_pytorch_vs_basic:.2f}x speedup")
    print(f"PyTorch vs Tiled: {speedup_pytorch_vs_tiled:.2f}x speedup")
    
    # Verify correctness by comparing results
    print(f"\n{'='*70}")
    print("Correctness Verification:")
    print(f"{'='*70}")
    
    max_diff_basic = torch.max(torch.abs(result_basic - result_pytorch)).item()
    max_diff_tiled = torch.max(torch.abs(result_tiled - result_pytorch)).item()
    
    print(f"Max difference (Basic vs PyTorch): {max_diff_basic:.6e}")
    print(f"Max difference (Tiled vs PyTorch): {max_diff_tiled:.6e}")
    
    # Small differences are expected due to floating point arithmetic
    if max_diff_basic < 1e-3 and max_diff_tiled < 1e-3:
        print("✓ All implementations produce correct results!")
    else:
        print("⚠ Warning: Results differ significantly. Check implementation.")
    
    print(f"\n{'='*70}\n")
    
    return {
        'size': size,
        'time_basic': time_basic,
        'time_tiled': time_tiled,
        'time_pytorch': time_pytorch,
        'speedup_tiled_vs_basic': speedup_tiled_vs_basic,
        'speedup_pytorch_vs_tiled': speedup_pytorch_vs_tiled
    }

# Run profiling for different matrix sizes
if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.")
        exit(1)
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Profile different matrix sizes to see how performance scales
    sizes = [512, 1024, 2048]
    results = []
    
    for size in sizes:
        result = profile_matrix_multiplication(size)
        results.append(result)
    
    # Summary table
    print("\n" + "="*70)
    print("Summary Across Different Matrix Sizes")
    print("="*70)
    print(f"{'Size':<10} {'Basic (ms)':<15} {'Tiled (ms)':<15} {'PyTorch (ms)':<15}")
    print("-"*70)
    
    for r in results:
        print(f"{r['size']:<10} {r['time_basic']*1000:<15.4f} "
              f"{r['time_tiled']*1000:<15.4f} {r['time_pytorch']*1000:<15.4f}")
    
    print("="*70)