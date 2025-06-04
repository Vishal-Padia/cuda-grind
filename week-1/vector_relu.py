import time
import torch
import triton
import triton.language as tl

DEVICE = "cuda:0"
print(f"Using device: {DEVICE}")

@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # get the current thread index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(y_ptr + offsets, y, mask=mask)

def relu(x):
    output = torch.empty_like(x)
    n_elements = output.numel()
    def grid(meta): return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

# Test the implementations
size = 98432
x = torch.rand(size, device=DEVICE)
x_cpu = x.cpu().numpy()

# Warm up everything
_ = relu(x)
_ = torch.relu(x)
_ = x_cpu * (x_cpu > 0)  # Proper numpy ReLU
torch.cuda.synchronize()

# Timing tests
start_time_torch = time.time()
output_torch = torch.relu(x)
torch.cuda.synchronize()
end_time_torch = time.time()

start_time_triton = time.time()
output_triton = relu(x)
torch.cuda.synchronize()
end_time_triton = time.time()

start_time_numpy = time.time()
output_numpy = x_cpu * (x_cpu > 0)  # Proper numpy ReLU
end_time_numpy = time.time()

print("=====TORCH=====")
print(output_torch)
print(f"Time taken by torch: {end_time_torch - start_time_torch} seconds")
print("=====TRITON=====")
print(output_triton)
print(f"Time taken by triton: {end_time_triton - start_time_triton} seconds")
print("=====NUMPY=====")
print(output_numpy)
print(f"Time taken by numpy: {end_time_numpy - start_time_numpy} seconds")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(8, 20)],
        line_arg="provider",
        line_vals=["torch", "triton", "numpy"],
        line_names=["Torch", "Triton", "Numpy"],
        ylabel="Time (ms)",
        plot_name="relu-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE)
    x_cpu = x.cpu().numpy()
    
    # Warm up
    if provider == "torch":
        torch.relu(x)
    elif provider == "triton":
        relu(x)
    elif provider == "numpy":
        x_cpu * (x_cpu > 0)
    
    torch.cuda.synchronize()
    
    # Measure time
    if provider == "torch":
        start_time = time.time()
        torch.relu(x)
        torch.cuda.synchronize()
        end_time = time.time()
    elif provider == "triton":
        start_time = time.time()
        relu(x)
        torch.cuda.synchronize()
        end_time = time.time()
    elif provider == "numpy":
        start_time = time.time()
        result = x_cpu * (x_cpu > 0)
        end_time = time.time()
    
    return (end_time - start_time) * 1000  # Convert to milliseconds

# Run the benchmark
benchmark.run(show_plots=True, print_data=True)