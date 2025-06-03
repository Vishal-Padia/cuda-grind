import time
import torch
import numpy as np
import triton
import triton.language as tl
from array import array

# get the device
DEVICE = torch.device("cuda:0")
print(f"Using device: {DEVICE}")


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # identifying the program
    pid = tl.program_id(axis=0) # gets the index of the current thread block in the first dimension
    block_start = pid * BLOCK_SIZE # calculates the starting index of the block

    # creates an array of indices for this thread block, it's similar to threadIdx.x in CUDA
    # creates a sequence of indices from 0 to BLOCK_SIZE - 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # creates a boolean mask to handle cases where the last block might be partially filled, 
    # this prevents out-of-bounds access when n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements 

    # loads the data from GPU memory into registers
    # mask ensures we only load valid elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # performs the acutal addition
    output = x + y

    # stores the result back to GPU memory
    # mask ensures we only write to valid positions
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y):
    output = torch.empty_like(x)

    # numel() gets the total number of elements
    n_elements = output.numel()

    # basically calculates the number of blocks needed
    # triton.cdiv is a helper function that calculates the ceiling division
    # meta['BLOCK_SIZE'] is gets teh block size from the metadata
    def grid(meta): return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # launches the kernel with specified block size
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)

# for numpy
x_cpu = x.cpu().numpy()
y_cpu = y.cpu().numpy()

# warm up everything
_ = add(x, y)
_ = x + y
_ = x_cpu + y_cpu

start_time_torch = time.time()
output_torch = x + y
end_time_torch = time.time()

start_time_triton = time.time()
output_triton = add(x, y)
end_time_triton = time.time()

start_time_numpy = time.time()
output_numpy = x_cpu + y_cpu
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
