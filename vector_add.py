import torch

import triton
import triton.language as tl

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = torch.device("cuda:0")
print(f"Using device: {DEVICE}")


@triton.jit
def add_kernel(
    x_ptr,  # pointer to first input vector
    y_ptr,  # pointer to second input vector
    output_ptr,  # pointer to output vector
    n_elements,  # size of the vector
    BLOCK_SIZE: tl.constexpr,  # number of elements each program should process
):
    # identifying the program
    pid = tl.program_id(axis=0)  # we use a 1D launch grid so axis is 0
    # program would process inputs that are offset from the initial data
    # suppose we have a vector of length 256 and block_size of 64, the program would each access the elements
    # [0:64, 64:128, 128:192, 192:256]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements
    # load x and y from DRAM, masking extra elements in case the input is not a multiple of the block size
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # we use a 1D grid where the size is the number of blocks
    def grid(meta): return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # we return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point
    return output


torch.manual_seed(0)
size = 98432

x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_triton)
print(output_torch)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# Benchmarking
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # argument names to use as an X-axis for the plot
        # Different possible values for 'x_name'
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        # Argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        line_vals=['triton', 'torch'],  # possible values for 'line_arg'
        line_names=['Triton', 'Torch'],  # label names for line
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel='GB/s',
        plot_name='vector-addition-performance',
        args={},  # values from function arguements not in x_names and y_name
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x+y, quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles)

    def gbps(ms): return 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)


benchmark.run(print_data=True, show_plots=True, save_path='vector_add.png')
