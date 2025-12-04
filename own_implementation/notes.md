### Plan:

So I was facing a problem where I was able to understand the kernel implementation but I wasn't able to implement it. Meaning it might by my C++ skills or it might be that I haven't genuinely understand the kernel implementation and it's optimizations and why those optimizations. So when I told Sriram this, he suggested that I go through the blog again and then understand algorithm and try to implement it on my own, I won't be able to ge the perfect implementation and I'll run into multiple bugs, and debuging those things would help strengthen my C++ skills and also I will understand each and every optimization quite better. 

"I do not understand what I can not create" -- Richard Feynman

Doing the above thing will probably take me around 15days or a month, but I think this is time well spent, understanding each optimizations and why those optimizations. Once I have a working version of the kernel I can benchmark it against LeiMao's implementation and check if there's any difference in my implementation and his. Also claude told me that this is the best advice someone can give, also before benchmarking the kernel try to profile the kernel using `nsys` or `ncu`. This will help me understand what's lacking in the kernel, meaning is it not parallel enough? does it have poor memory access pattern? etc. So this will help me in deriving my intuition regarding what can be optimized more.

So yes, I'll be doing this for the next 15 days or a month. It will be slow but I'll learn a lot.

Nsys commands to profile the kernel:
```bash
nsys profile --stats=True ./profile_cuda_gemm_fp16
```

```bash
nsys profile --stats=True ./profile_cuda_gemm_fp32
```

Basic info about nsys:
'osrt_sum' - OS Runtime Statistics

This is CPU-side profiling. It shows where your CPU spent time while the GPU was working:

- Time (%): Percentage of total CPU time spent in this system call
- Total Time (ns): Aggregate nanoseconds across all calls
- Num Calls: How many times this function was called
- Avg/Med/Min/Max: Statistics on individual call duration
- StdDev: Variance in call duration

'cuda_api_sum' - CUDA API Statistics

'cuda_gpu_kern_sum' - GPU Kernel Execution Statistics

'cuda_gpu_mem_time_sum' - Memory Operation Timing

'cuda_gpu_mem_size_sum' - Memory Volume

### Here's a decision tree Claude gave me in order to decide what to optimize first:
```
Is bandwidth utilization < 10%?
  ├─ YES → Memory access pattern is the problem
  │         → Coalesce, improve spatial locality, reduce scattered access
  │         → Your situation
  │
  └─ NO → Is it 10-50%?
          ├─ YES → Mixed problem, profile deeper
          │         → Check cache hit rates with nsys --stats=full
          │         → Look at branch divergence
          │
          └─ NO (> 50%) → Compute is the bottleneck
                          → Reduce instruction count
                          → Improve ILP (instruction-level parallelism)
                          → Better algorithm complexity

Is memory copy time > 30% of total?
  ├─ YES → Reduce data movement
  │         → Batch operations
  │         → Keep data on GPU
  │
  └─ NO → Focus elsewhere

Is API overhead > 5% of total?
  ├─ YES → Reduce kernel launch overhead
  │         → Batch kernel calls
  │         → Use persistent kernels
  │
  └─ NO → Kernel execution is your target
```

# 00: Non-Coalesced Memory Access Implementation

### 01: Profiling the kernel:

Profiling the fp16 kernel:
- 48% of the CPU time is spent in cudaStreamSynchronize (waiting for my slow kernel to finish)
- 23.5% of time is spent while CPU is waiting for GPU events (cudaEventSynchronize)
- 10.2% of time is spend in creating a new asynchronous stream (cudaStreamCreate)

Profiling the fp32 kernel:
- 37.6% of CPU time is spent here = CPU waiting for my slow kernel to finish. (cudaStreamSynchronize)
- 24% of time is spent in allocating pinned host memory. (cudaHostAlloc)
- 18.4% of time is spent while CPU is waiting for GPU events (cudaEventSynchronize)


### 02: Benchmarking the kernel:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.5797 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Non-Coalesced Memory Access Implementation
cuBLAS GEMM Kernel Performance
Latency: 0.672896 ms
Effective Bandwidth: 299.194 GB/s
Effective TFLOPS: 204.25 TFLOPS
Custom GEMM Kernel Performance
Latency: 320.735 ms
Effective Bandwidth: 0.627705 GB/s
Effective TFLOPS: 0.428513 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 0.209798%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.5797 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Non-Coalesced Memory Access Implementation
cuBLAS GEMM Kernel Performance
Latency: 3.46922 ms
Effective Bandwidth: 58.0323 GB/s
Effective TFLOPS: 39.6167 TFLOPS
Custom GEMM Kernel Performance
Latency: 319.431 ms
Effective Bandwidth: 0.630266 GB/s
Effective TFLOPS: 0.430262 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.08606%
```

### 03: Comparing the results with LeiMao's implementation:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.5797 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V00
cuBLAS GEMM Kernel Performance
Latency: 0.657216 ms
Effective Bandwidth: 306.332 GB/s
Effective TFLOPS: 209.123 TFLOPS
Custom GEMM Kernel Performance
Latency: 319.888 ms
Effective Bandwidth: 0.629365 GB/s
Effective TFLOPS: 0.429647 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 0.205452%
```
FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.5797 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V00
cuBLAS GEMM Kernel Performance
Latency: 3.4601 ms
Effective Bandwidth: 58.1853 GB/s
Effective TFLOPS: 39.7211 TFLOPS
Custom GEMM Kernel Performance
Latency: 319.82 ms
Effective Bandwidth: 0.6295 GB/s
Effective TFLOPS: 0.429739 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.08189%
```
FP16:

| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 320.735 ms | Latency: 319.888 ms |
| Bandwidth: 0.627705 GB/s | Bandwidth: 0.629365 GB/s |
| TFLOPS: 0.429647 TFLOPS | TFLOPS: 0.429739 TFLOPS |

FP32:

| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 319.431 ms | Latency: 319.82 ms |
| Bandwidth: 0.630266 GB/s | Bandwidth: 0.6295 GB/s |
| TFLOPS: 0.430262 TFLOPS | TFLOPS: 0.429739 TFLOPS |

If you see there's not a lot of difference between LeiMao's implementation and my implementation.  

### 04: What can be optimized more?

- Changing the memory access pattern to coalesced memory access.
- Using Tiling techniques to improve the memory access pattern.
- Using Warp Tiling techniques to improve the memory access pattern.

# 01: Coalesced Memory Access Implementation

### 01: Profiling the kernel:

Profiling the Fp16 kernel:
- 28.7% percent of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 22.6% percent of the time went in cudaHostAlloc (allocating pinned host memory)
- 22.0% percent of the time went in cudaStreamSynchronize (waiting for the kernel to finish)
- 11.3% percent of the time went in cudaEventSynchronize (waiting for the kernel to finish)

Profiling the Fp32 kernel:
- 31.8% percent of the time went in cudaHostAlloc (allocating pinned host memory)
- 21.6% percent of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 17.9% percent of the time went in cudaStreamSynchronize (waiting for the kernel to finish)
- 11.7% percent of the time went in cudaFreeHost (freeing pinned host memory)
- 9.1% percent of the time went in cudaEventSynchronize (waiting for the kernel to finish)

### 02: Benchmarking the kernel:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.5685 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Coalesced Memory Access Implementation
cuBLAS GEMM Kernel Performance
Latency: 0.72464 ms
Effective Bandwidth: 277.83 GB/s
Effective TFLOPS: 189.665 TFLOPS
Custom GEMM Kernel Performance
Latency: 42.79 ms
Effective Bandwidth: 4.70499 GB/s
Effective TFLOPS: 3.21194 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.69348%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.5685 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Coalesced Memory Access Implementation
cuBLAS GEMM Kernel Performance
Latency: 3.80214 ms
Effective Bandwidth: 52.9508 GB/s
Effective TFLOPS: 36.1478 TFLOPS
Custom GEMM Kernel Performance
Latency: 47.548 ms
Effective Bandwidth: 4.23418 GB/s
Effective TFLOPS: 2.89053 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 7.99643%
```

### 03: Comparing the results with LeiMao's implementation:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.5685 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V01
cuBLAS GEMM Kernel Performance
Latency: 0.718848 ms
Effective Bandwidth: 280.068 GB/s
Effective TFLOPS: 191.193 TFLOPS
Custom GEMM Kernel Performance
Latency: 42.7653 ms
Effective Bandwidth: 4.70771 GB/s
Effective TFLOPS: 3.2138 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.68091%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.5685 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V01
cuBLAS GEMM Kernel Performance
Latency: 3.45392 ms
Effective Bandwidth: 58.2893 GB/s
Effective TFLOPS: 39.7922 TFLOPS
Custom GEMM Kernel Performance
Latency: 44.4723 ms
Effective Bandwidth: 4.52701 GB/s
Effective TFLOPS: 3.09044 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 7.76645%
```

FP16:
| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 42.79 ms | Latency: 42.7653 ms |
| Bandwidth: 4.70499 GB/s | Bandwidth: 4.70771 GB/s |
| TFLOPS: 3.21194 TFLOPS | TFLOPS: 3.2138 TFLOPS |

FP32:

| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 47.548 ms | Latency: 44.4723 ms |
| Bandwidth: 4.23418 GB/s | Bandwidth: 4.52701 GB/s |
| TFLOPS: 2.89053 TFLOPS | TFLOPS: 3.09044 TFLOPS |

If you see that there's not a lot of difference between LeiMao's implementation and my implementation. Apart from the TFLOPS in FP32, my implementation is slightly bad than LeiMao's implementation.

### 04: What can be optimized more?

- We can use 2d block tiling to improve the memory access pattern.
- Use warp level tiling
- Use thread level tiling
 and much more

# 02: 2D Block Tiling Implementation

### 01: Profiling the kernel:

Fp16:
- 34.5% of the time went in cudaHostAlloc (allocating pinned host memory)
- 22.8% of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 14.4% of the time went in cudaStreamSynchronize (waiting for the kernel to finish)
- 12.7% of the time went in cudaFreeHost (freeing pinned host memory)

Fp32:
- 35.4% of the time went in cudaHostAlloc (allocating pinned host memory)
- 21.2% of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 14.6% of the time went in cudaStreamSynchronize (waiting for the kernel to finish)
- 12.8% of the time went in cudaFreeHost (freeing pinned host memory)

### 02: Benchmarking the kernel:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

2D block tiling Implementation
cuBLAS GEMM Kernel Performance
Latency: 3.80621 ms
Effective Bandwidth: 52.8943 GB/s
Effective TFLOPS: 36.1092 TFLOPS
Custom GEMM Kernel Performance
Latency: 35.1448 ms
Effective Bandwidth: 5.72848 GB/s
Effective TFLOPS: 3.91064 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 10.8301%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

2D block tiling Implementation
cuBLAS GEMM Kernel Performance
Latency: 3.80554 ms
Effective Bandwidth: 52.9036 GB/s
Effective TFLOPS: 36.1155 TFLOPS
Custom GEMM Kernel Performance
Latency: 35.1404 ms
Effective Bandwidth: 5.7292 GB/s
Effective TFLOPS: 3.91114 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 10.8295%
```

### 03: Comparing the results with LeiMao's implementation:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V02
cuBLAS GEMM Kernel Performance
Latency: 0.71744 ms
Effective Bandwidth: 280.618 GB/s
Effective TFLOPS: 191.569 TFLOPS
Custom GEMM Kernel Performance
Latency: 34.9449 ms
Effective Bandwidth: 5.76126 GB/s
Effective TFLOPS: 3.93302 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 2.05306%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V02
cuBLAS GEMM Kernel Performance
Latency: 3.79997 ms
Effective Bandwidth: 52.9811 GB/s
Effective TFLOPS: 36.1684 TFLOPS
Custom GEMM Kernel Performance
Latency: 35.1519 ms
Effective Bandwidth: 5.72734 GB/s
Effective TFLOPS: 3.90986 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 10.8101%
```

FP16:
| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 35.1448 ms | Latency: 34.9449 ms |
| Bandwidth: 5.72848 GB/s | Bandwidth: 5.76126 GB/s |
| TFLOPS: 3.91064 TFLOPS | TFLOPS: 3.93302 TFLOPS |

FP32:
| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 35.1404 ms | Latency: 35.1519 ms |
| Bandwidth: 5.7292 GB/s | Bandwidth: 5.72734 GB/s |
| TFLOPS: 3.91114 TFLOPS | TFLOPS: 3.90986 TFLOPS |

There's not a lot of difference between LeiMao's implementation and my implementation, because well I learnt from his implementation :p

### 04: What can be optimized more?

- Have better tiling strategy.
- Use better memory access pattern.
- Use wmma to improve the performance.

# 03: 2D Block Tiling and 1D Thread Tiling Implementation

### 01: Profiling the kernel:

Fp16:
- 41.1% of the time went in cudaHostAlloc (allocating pinned host memory)
- 28.1% of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 11.0% of the time went in cudaStreamSynchronize (waiting for the kernel to finish)
- 8.2% of the time went in cudaFreeHost (freeing pinned host memory)

Fp32:
- 33.0% of the time went in cudaHostAlloc (allocating pinned host memory)
- 30.9% of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 11.8% of the time went in cudaFreeHost (freeing pinned host memory)
- 8.5% of the time went in cudaMalloc (allocating device memory)

### 02: Benchmarking the kernel:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

2D block tiling & 1D thread Tiling Implementation
cuBLAS GEMM Kernel Performance
Latency: 0.712416 ms
Effective Bandwidth: 282.597 GB/s
Effective TFLOPS: 192.92 TFLOPS
Custom GEMM Kernel Performance
Latency: 22.2236 ms
Effective Bandwidth: 9.05915 GB/s
Effective TFLOPS: 6.18438 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 3.20568%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

2D Block Tiling and 1D Thread Tiling Implementation
cuBLAS GEMM Kernel Performance
Latency: 3.73536 ms
Effective Bandwidth: 53.8975 GB/s
Effective TFLOPS: 36.794 TFLOPS
Custom GEMM Kernel Performance
Latency: 10.9152 ms
Effective Bandwidth: 18.4447 GB/s
Effective TFLOPS: 12.5916 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 34.2217%
```

### 03: Comparing the results with LeiMao's implementation:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V03
cuBLAS GEMM Kernel Performance
Latency: 0.705536 ms
Effective Bandwidth: 285.353 GB/s
Effective TFLOPS: 194.801 TFLOPS
Custom GEMM Kernel Performance
Latency: 22.2361 ms
Effective Bandwidth: 9.05404 GB/s
Effective TFLOPS: 6.18089 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 3.17293%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V03
cuBLAS GEMM Kernel Performance
Latency: 3.72707 ms
Effective Bandwidth: 54.0174 GB/s
Effective TFLOPS: 36.8759 TFLOPS
Custom GEMM Kernel Performance
Latency: 10.8972 ms
Effective Bandwidth: 18.4751 GB/s
Effective TFLOPS: 12.6123 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 34.2022%
```

FP16:
| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 22.2236 ms | Latency: 22.2361 ms |
| Bandwidth: 9.05915 GB/s | Bandwidth: 9.05404 GB/s |
| TFLOPS: 6.18438 TFLOPS | TFLOPS: 6.18089 TFLOPS |

FP32:
| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 10.9152 ms | Latency: 10.8972 ms |
| Bandwidth: 18.4447 GB/s | Bandwidth: 18.4751 GB/s |
| TFLOPS: 12.5916 TFLOPS | TFLOPS: 12.6123 TFLOPS |

### 04: What can be optimized more?

- Use better tiling strategy.
- Use better memory access pattern.
- Use wmma to improve the performance.

# 04: 2D Block Tiling and 2D Thread Tiling Implementation

### 01: Profiling the kernel:

Fp16:
- 73.1% of the time went in cudaLaunchKernel (launching the kernel)
- 9.4% of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 7.5% of the time went in cudaHostAlloc (allocating pinned host memory)
- 3.5% of the time went in cudaStreamSynchronize (waiting for the kernel to finish)

Fp32:
- 41.7% of the time went in cudaHostAlloc (allocating pinned host memory)
- 26.3% of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 14.8% of the time went in cudaFreeHost (freeing pinned host memory)
- 4.4% of the timt went in cudaMemcpy (copying data from host to device)

### 02: Benchmarking the kernel:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

2D block tiling & 2D thread Tiling Implementation
cuBLAS GEMM Kernel Performance
Latency: 0.711104 ms
Effective Bandwidth: 283.118 GB/s
Effective TFLOPS: 193.275 TFLOPS
Custom GEMM Kernel Performance
Latency: 23.128 ms
Effective Bandwidth: 8.70489 GB/s
Effective TFLOPS: 5.94253 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 3.07465%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

2D block tiling & 2D thread Tiling Implementation
cuBLAS GEMM Kernel Performance
Latency: 3.73904 ms
Effective Bandwidth: 53.8445 GB/s
Effective TFLOPS: 36.7578 TFLOPS
Custom GEMM Kernel Performance
Latency: 6.62883 ms
Effective Bandwidth: 30.3714 GB/s
Effective TFLOPS: 20.7335 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 56.4057%
```

### 03: Comparing the results with LeiMao's implementation:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V04
cuBLAS GEMM Kernel Performance
Latency: 0.704928 ms
Effective Bandwidth: 285.599 GB/s
Effective TFLOPS: 194.969 TFLOPS
Custom GEMM Kernel Performance
Latency: 23.1535 ms
Effective Bandwidth: 8.6953 GB/s
Effective TFLOPS: 5.93599 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 3.04458%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V04
cuBLAS GEMM Kernel Performance
Latency: 3.74038 ms
Effective Bandwidth: 53.8251 GB/s
Effective TFLOPS: 36.7446 TFLOPS
Custom GEMM Kernel Performance
Latency: 6.72154 ms
Effective Bandwidth: 29.9525 GB/s
Effective TFLOPS: 20.4476 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 55.6478%
```

FP16:
| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 23.128 ms | Latency: 23.1535 ms |
| Bandwidth: 8.70489 GB/s | Bandwidth: 8.6953 GB/s |
| TFLOPS: 5.94253 TFLOPS | TFLOPS: 5.93599 TFLOPS |

FP32:
| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 6.20898 ms | Latency: 6.72154 ms |
| Bandwidth: 30.3714 GB/s | Bandwidth: 29.9525 GB/s |
| TFLOPS: 20.7335 TFLOPS | TFLOPS: 20.4476 TFLOPS |

### 04: What can be optimized more?

- Vectorized memory access
- Warp Tiling
- Transpose the matrix A in the shared memory
- WMMA implementation

# 05: 2D Block Tiling, 2D Thread Tiling, with Vectorized Memory Access Implementation

### 01: Profiling the kernel:

Fp16:
- 83.4% of the time went in cudaLaunchKernel (launching the kernel)
- 5.8% of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 4.9% of the time went in cudaHostAlloc (allocating pinned host memory)
- 1.9% of the time went in cudaStreamSynchronize (waiting for the kernel to finish)

Fp32:
- 41.1% of the time went in cudaHostAlloc (allocating pinned host memory)
- 27.1% of the time went in cudaStreamCreate (allocating a new asynchronous stream)
- 14.8% of the time went in cudaFreeHost (freeing pinned host memory)
- 4.5% of the time went in cudaMemcpy (copying data from host to device)

### 02: Benchmarking the kernel:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

2D block tiling & 2D thread with Vectorized Memory Access Implementation
cuBLAS GEMM Kernel Performance
Latency: 0.724512 ms
Effective Bandwidth: 277.879 GB/s
Effective TFLOPS: 189.699 TFLOPS
Custom GEMM Kernel Performance
Latency: 17.4932 ms
Effective Bandwidth: 11.5089 GB/s
Effective TFLOPS: 7.85671 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 4.14168%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

2D block tiling & 2D thread with Vectorized Memory Access Implementation
cuBLAS GEMM Kernel Performance
Latency: 3.81859 ms
Effective Bandwidth: 52.7227 GB/s
Effective TFLOPS: 35.9921 TFLOPS
Custom GEMM Kernel Performance
Latency: 4.74579 ms
Effective Bandwidth: 42.4221 GB/s
Effective TFLOPS: 28.9602 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 80.4627%
```

### 03: Comparing the results with LeiMao's implementation:

FP16:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V05 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 0.716512 ms
Effective Bandwidth: 280.981 GB/s
Effective TFLOPS: 191.817 TFLOPS
Custom GEMM Kernel Performance
Latency: 17.4907 ms
Effective Bandwidth: 11.5105 GB/s
Effective TFLOPS: 7.85785 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 4.09654%
```

FP32:
```bash
Device Name: NVIDIA GeForce RTX 4080 SUPER
Memory Size: 15.57 GB
Peak Bandwitdh: 736.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V05 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 3.79782 ms
Effective Bandwidth: 53.011 GB/s
Effective TFLOPS: 36.1889 TFLOPS
Custom GEMM Kernel Performance
Latency: 4.74112 ms
Effective Bandwidth: 42.4639 GB/s
Effective TFLOPS: 28.9887 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 80.1039%
```

FP16:
| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 17.4932 ms | Latency: 17.4907 ms |
| Bandwidth: 11.5089 GB/s | Bandwidth: 11.5105 GB/s |
| TFLOPS: 7.85671 TFLOPS | TFLOPS: 7.85785 TFLOPS |

FP32:
| My Implementation | LeiMao's Implementation |
|------------------|-------------------------|
| Latency: 4.74112 ms | Latency: 4.74112 ms |
| Bandwidth: 42.4639 GB/s | Bandwidth: 42.4639 GB/s |
| TFLOPS: 28.9887 TFLOPS | TFLOPS: 28.9887 TFLOPS |

### 04: What can be optimized more?

- Warp Tiling
- WMMA implementation

# 06: 2D Block Tiling and 2D Warp Tiling and 2D Thread Tiling and Vectorized Memory Access

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?

# 07: 2D Block Tiling, 2D Warp Tiling, 2D Thread Tiling, Matrix Transpose with Vectorized Memory Access, and WMMA Implementation

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?
