import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    mat1,
    mat2,
    result,
    M,
    N,
    K,
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
):
    # 2D block decomposition using program_id
    row_block_id = tl.program_id(0)
    col_block_id = tl.program_id(1)

    # calculate row/column indices for this block
    row_start = row_block_id * ROW_BLOCK_SIZE
    col_start = col_block_id * COL_BLOCK_SIZE

    row_offsets = row_start + tl.arange(0, ROW_BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, COL_BLOCK_SIZE)

    # initialize accumulator
    accumulator = tl.zeros((ROW_BLOCK_SIZE, COL_BLOCK_SIZE), dtype=tl.float32)

    # loop over K dimension in block
    for k in range(0, K, K_SIZE):
        # create masks for boundary checks
        k_offsets = k + tl.arange(0, K_SIZE)
        a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None]< K) & (col_offsets[None, :] < N)

        # load blocks from global memory
        a = tl.load(mat1 + row_offsets[:, None] * K + k_offsets[None, :], mask=a_mask, other=0.0)
        b = tl.load(mat2 + k_offsets[:, None] * N + col_offsets[None, :], mask=b_mask, other=0.0)

        # compute partial matrix multiplication
        accumulator += tl.dot(a, b)

    # create output mask and store result
    c_mask = (row_offsets[:, None] < M) & (col_offsets[None, :]< N)
    tl.store(result + row_offsets[:, None] * N + col_offsets[None, :], accumulator, mask=c_mask)

def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device)

    ROW_BLOCK = 64
    COL_BLOCK = 64
    K_BLOCK = 64

    grid = lambda META: (triton.cdiv(M, META['ROW_BLOCK_SIZE']), triton.cdiv(N, META['COL_BLOCK_SIZE']))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        ROW_BLOCK_SIZE=ROW_BLOCK,
        COL_BLOCK_SIZE=COL_BLOCK,
        K_SIZE=K_BLOCK
    )

    return c

# Benchmark execution time and memory bandwidth
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Dimension names
        x_vals=[128 * i for i in range(2, 33)],  # Test different matrix sizes
        line_arg='provider', 
        line_vals=['torch', 'triton'],
        line_names=['Torch', 'Triton'],
        ylabel='TFLOPS',  # We'll measure TFLOPS
        plot_name='matmul-performance',
        args={},
    )
)
def benchmark(M, N, K, provider):
    # Matrix dimensions need to be powers of 2
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    
    # Warmup
    if provider == 'torch':
        torch.matmul(a, b)
    else:
        matmul(a, b)
    
    # Synchronize GPU
    torch.cuda.synchronize()
    
    # Calculate theoretical FLOPs for matmul
    flops = 2 * M * N * K  # Each element needs K multiplications and K-1 additions
    
    # Measure execution time
    if provider == 'torch':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):  # Multiple runs for more stable measurements
            torch.matmul(a, b)
        end.record()
        
        # Synchronize and get time in seconds
        end.synchronize()
        time_ms = start.elapsed_time(end) / 10  # Average over 10 runs
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            matmul(a, b)
        end.record()
        
        end.synchronize()
        time_ms = start.elapsed_time(end) / 10
    
    # Return TFLOPS
    return flops / (time_ms * 1e6)  # Convert ms to seconds and to TFLOPS    

# Run benchmark    
benchmark.run(show_plots=True, print_data=True, save_path="matmul")