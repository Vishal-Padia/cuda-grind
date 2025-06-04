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

size = 98432
x = torch.rand(size, device=DEVICE)

# for numpy
x_cpu = x.cpu().numpy()

# warm up everything
_ = relu(x)
_ = x * 0

start_time_torch = time.time()
output_torch = torch.relu(x)
end_time_torch = time.time()

start_time_triton = time.time()
output_triton = relu(x)
end_time_triton = time.time()

start_time_numpy = time.time()
output_numpy = x_cpu * 0
end_time_numpy = time.time()

print("=====TORCH=====")
print(output_torch)
print(f"Time taken by torch: {end_time_torch - start_time_torch} seconds")

print("=====TRITON=====")
print(output_triton)
print(f"Time taken by triton {end_time_triton - start_time_triton} seconds")

print("=====NUMPY=====")
print(output_numpy)
print(f"Time taken by numpy: {end_time_numpy - start_time_numpy} seconds")


    