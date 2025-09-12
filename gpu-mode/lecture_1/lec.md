# Lecture 1

The goal of this is to learn more about cuda.

Currently the GPU-Mode lectures goes through the textbook: Programming Massively Parallel Processors. They like go through a couple of chapters per lecture, and explain everything in depth.

Goal of this Lecture
1. Integrate a CUDA kernel inside a pytorch program
2. Learn how to profile it

Most of the code is here https://github.com/msaroufim/cudamodelecture1

Pointwise ops are super important in ML/DL (softmax and others).

CUDA is async. So if we were to use python's time module we measure the time taken to launch kernel ie the overhead and not the actual time.

We can setup a start event and end event using `torch.cuda.Event` to measure the kernel's time. We also need to warmup the operations before running to reduce the overhead of kernel starting up.

The host uses autograd profiler to profile the operations.

PyTorch profiler gives a chrome trace, which is basically a pretty UI showing the time, memory, and stuff.

PyTorch repo has cuda kernel which it calls in it's operations.

CUDA is usually written in C/C++.

Here's how to load cuda kernel in torch:
```
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_worl() {
    return "Hello World";
}
"""
my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True
)

print(my_module.hello_world())
```

the cpp_source can also be a paths to file/files.
it will create pybind, and `build.ninja` which is basically a "cmake" file.

This stuff are auto-generated when we run the above code.

Above is the basic thing.

We can write proper cuda in the same way.
```
import torch
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and C++ wrapper
cuda_source = '''
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);

    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

    return result;
    }
'''

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

# Load the CUDA kernel as a PyTorch extension
square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./load_inline_cuda',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix(a))
```

The above snippet is from here: https://github.com/gpu-mode/profiling-cuda-in-torch/blob/main/load_inline.py

In the load inline you'll see that we are passing the cpp_source which is just calling the function and in the cuda source which is the actual cuda kernel which we have wrote.

The `02` in `extra_cuda_cflags` is just an compiler optimization flag, there are more flags like this.

There's also numba which is cuda code in python (sort off), it has boiler plate code which is different but it's cuda.

There's also Triton which is a python-wrapper (kind off), it doesn't generate `.cu` ie cuda files, it generates ptx. Also using triton is super simple it's basically python, so we don't need to use `load_inline` or something we can easily call the function like we do in python.

When using Triton we need to make sure the `block_size` is correct, else the performance would not be better than vanilla torch operations.

Triton also has a debugger `triton.jit(interpret=True)` and then we can add breakpoints to see what's going ie debugging. Almost everything ia a WrappedTensor so inspect variables with `var_name.tensor`

You can't add print statements in the function wraper under `triton.jit()`, by doing so the kernel will crash, that's why they created the debugger for triton.

Cheat: Generate a triton kernel
```
import torch

torch.compile(torch.square)
```
run this using `TORCH_LOGS="output_code" python compile_square.py`

this will generate actual triton code for this kernel.

This can be used when you don't want to write triton kernel, you can use this as a starting point to start optimizing your kernel.

^Here they don't operate row-by-row like a normal triton kernel, and it also passes some compiler heuristic things.


ncu profiler: this is the most useful cuda profiler

`ncu python train.py`, most cloudprovider don't provide this but can be installed and used.

It also provides "comments" basically small snippets telling that "hey this is the peak and you're only at 60%" this will show you that can optimize more. Bascilly contains actionable hints. For exmaple:
```
OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.4 full waves across all SMs. Look at Launch Statistics for more details.
```

This will help us understand how can we optimize more or what we are doing wrong and stuff.

It also have a really good visual profiler
`ncu --set full -o output $(which python) train.py`

Once you work on those actionable insights, there's a high chance you're kernel will be more optimized than previous version.

The visual profiler will show the memory usage and stuff for your code line-by-line, you can change those lines to see the full effect.

Advice:
Try to fuse kernels into a single one, meaning try to reduce the number of kernels (but don't fuse weird kernels lol).


