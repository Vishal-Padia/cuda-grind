This basically contains the notes I have taken while reading LeiMao's [blog](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/).

### 00: Naive Implementation with Non-Coalesced Memory Access
This is same as the implementation I did, checkout [here](https://github.com/Vishal-Padia/cuda-grind/blob/main/pmpp/matrix_multiplication.cu). But the LeiMao's implementation is more concise, for example, my implementation considers only square matrices (WIDTHxWIDTH), also contains hardcoded indexing for square matrices, Lei Mao's blog has implementation which contains non-square matrices: `m x k` times `k x n` = `m x n`. It also contains leading dimensions since matrices can be stored as sub-matrices of larger arrays, it also has cublas sytle alpha/beta which is needed for fused operations. Overall we can say that LeiMao's implementation is more efficient and flexible than mine, and also his is production ready for all datatype since he has used templates, but at it's core both are same.

FP16:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V00
cuBLAS GEMM Kernel Performance
Latency: 2.43456 ms
Effective Bandwidth: 82.6953 GB/s
Effective TFLOPS: 56.4533 TFLOPS
Custom GEMM Kernel Performance
Latency: 2212.64 ms
Effective Bandwidth: 0.0909892 GB/s
Effective TFLOPS: 0.0621153 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 0.110029%
```

FP32:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V00
cuBLAS GEMM Kernel Performance
Latency: 18.567 ms
Effective Bandwidth: 10.8432 GB/s
Effective TFLOPS: 7.40231 TFLOPS
Custom GEMM Kernel Performance
Latency: 2212.61 ms
Effective Bandwidth: 0.0909907 GB/s
Effective TFLOPS: 0.0621163 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 0.839148%
```

From the results we can see that the custom GEMM kernel is much slower than the cuBLAS GEMM kernel, this is because the we are compute bound meaning the T4 gpu is not able to keep up with the compute requirements of the custom GEMM Kernel. The T4 has Tensor Cores that deliver 65 TFLOPS for mixed FP16/FP32. cuBLAS achieves 56.45 TFLOPS (87% of peak).

### 01: Naive Implementation with Coalesced Memory Access

This is super similar to the previous implementation, the only difference is that we are accessing the memory in a coalesced manner, so basically we are accessing the memory in a row-wise manner instead of column-wise manner. Now because of this change, the threads in the same warp read the elements in the same row of B that is stored in row-major order, resulting in a coalesced memory access.

FP16:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V01
cuBLAS GEMM Kernel Performance
Latency: 2.45731 ms
Effective Bandwidth: 81.9296 GB/s
Effective TFLOPS: 55.9306 TFLOPS
Custom GEMM Kernel Performance
Latency: 205.784 ms
Effective Bandwidth: 0.978342 GB/s
Effective TFLOPS: 0.667881 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.19412%
```

FP32:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V01
cuBLAS GEMM Kernel Performance
Latency: 18.6422 ms
Effective Bandwidth: 10.7995 GB/s
Effective TFLOPS: 7.37247 TFLOPS
Custom GEMM Kernel Performance
Latency: 213.749 ms
Effective Bandwidth: 0.941883 GB/s
Effective TFLOPS: 0.642992 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 8.72153%
```

Comparing the Non-Naive and Naive implementation we already see there's a significant improvement in the performance, like let's take TFLOPS as an example, in Non-Naive FP16 we have 0.0621153 TFLOPS, FP32 we have 0.0621163 TFLOPS, in Naive FP16 we have 0.667881 TFLOPS, FP32 we have 0.642992 TFLOPS. Also the latency is significantly reduced in the Naive implementation. So just by switching memory access pattern from column-wise to row-wise, we are able to achieve a significant improvement in the performance.

### 02: Implementation with 2D Block Tiling
In the previous approach, we were computing the output element by reading an entire row of A and entire column of B from global memory, computing one number, then throwing away all that data and starting over for the next element. Multiple output elements share the input data, When computing output matrix C, notice that: Every element in row i of C needs the same row i from A and Every element in column j of C needs the same column j from B. So if we are computing a 2x2 block of C, all four elements need the same 2 rows from A and same 2 columns from B. Instead of fetching these rows/columns 4 times, fetch them once and reuse them.

fp16:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V02
cuBLAS GEMM Kernel Performance
Latency: 2.45318 ms
Effective Bandwidth: 82.0675 GB/s
Effective TFLOPS: 56.0247 TFLOPS
Custom GEMM Kernel Performance
Latency: 126.695 ms
Effective Bandwidth: 1.58906 GB/s
Effective TFLOPS: 1.0848 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.93628%
```

fp32:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V02
cuBLAS GEMM Kernel Performance
Latency: 18.3132 ms
Effective Bandwidth: 10.9935 GB/s
Effective TFLOPS: 7.50491 TFLOPS
Custom GEMM Kernel Performance
Latency: 137.395 ms
Effective Bandwidth: 1.46531 GB/s
Effective TFLOPS: 1.00032 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 13.3288%
```

Comparing this with the previous implementation, we can see that the performance has signifacntly improved, in FP16 we have 1.0848 TFLOPS, in FP32 we have 1.00032 TFLOPS. Also the latency is significantly reduced in the 2D Block Tiling implementation.

### 03: Implementation with 2D Block Tiling and 1D Thread Tiling
In this we cache some even smaller tiles of the input matrices to shared memory to the registers and to the threads. Each thread is responsible for computing small tile of output matrix D instead of one single element. Because the registers are the fastest to access, the performance of this implementation should be much better than the previous one.

fp16:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V03
cuBLAS GEMM Kernel Performance
Latency: 2.45123 ms
Effective Bandwidth: 82.1328 GB/s
Effective TFLOPS: 56.0693 TFLOPS
Custom GEMM Kernel Performance
Latency: 51.4458 ms
Effective Bandwidth: 3.91338 GB/s
Effective TFLOPS: 2.67153 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 4.76469%
```

fp32:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V03
cuBLAS GEMM Kernel Performance
Latency: 19.5491 ms
Effective Bandwidth: 10.2985 GB/s
Effective TFLOPS: 7.03044 TFLOPS
Custom GEMM Kernel Performance
Latency: 65.9443 ms
Effective Bandwidth: 3.05298 GB/s
Effective TFLOPS: 2.08417 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 29.6449%
```

From the above results we can see that the performance has signifacntly improved, in FP16 we have 2.67153 TFLOPS, in FP32 we have 2.08417 TFLOPS. Also the latency is significantly reduced in the 2D Block Tiling and 1D Thread Tiling implementation. This is because we are caching the tiles of the input matrices to the registers and to the threads, so we are not accessing the global memory as much, and also we are using the registers to store the values, so we are not accessing the shared memory as much. This is a trade-off between the performance and the memory bandwidth.

### 04: Implementation with 2D Block Tiling and 2D Thread Tiling
For each chunk of the K dimension:
    1. Load A and B tiles into shared memory (all threads help)
    2. Synchronize (make sure everyone loaded)
    3. Compute using the shared memory tiles (each thread does its piece)
    4. Synchronize (make sure everyone finished)

Each thread is doing the same computation pattern, just on different data indices.

fp16:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V04
cuBLAS GEMM Kernel Performance
Latency: 2.45923 ms
Effective Bandwidth: 81.8656 GB/s
Effective TFLOPS: 55.8869 TFLOPS
Custom GEMM Kernel Performance
Latency: 17.1763 ms
Effective Bandwidth: 11.7212 GB/s
Effective TFLOPS: 8.00168 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 14.3176%
```

fp32:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V04
cuBLAS GEMM Kernel Performance
Latency: 19.5892 ms
Effective Bandwidth: 10.2774 GB/s
Effective TFLOPS: 7.01604 TFLOPS
Custom GEMM Kernel Performance
Latency: 34.1946 ms
Effective Bandwidth: 5.88768 GB/s
Effective TFLOPS: 4.01932 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 57.2876%
```

From the above results we can see that the performance has signifacntly improved, in FP16 we have 8.00168 TFLOPS, in FP32 we have 4.01932 TFLOPS. Also the latency is significantly reduced in the 2D Block Tiling and 2D Thread Tiling implementation. This is because we are caching the tiles of the input matrices to the shared memory, so we are not accessing the global memory as much, and also we are using the shared memory to store the values, so we are not accessing the registers as much. This is a trade-off between the performance and the memory bandwidth.

### 05: Implementation with 2D Block Tiling, 2D Thread Tiling, and Matrix Transpose with Vectorized Memory Access

In this implementation, we are transposing the matrix A and B in the shared memory to make the memory access more coalesced. This is done by using the vectorized memory access to transpose the matrix A and B. We are also using the vectorized memory access to read the values from the shared memory and to write the values to the global memory.

fp16:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V05 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 2.45526 ms
Effective Bandwidth: 81.9979 GB/s
Effective TFLOPS: 55.9773 TFLOPS
Custom GEMM Kernel Performance
Latency: 12.126 ms
Effective Bandwidth: 16.6029 GB/s
Effective TFLOPS: 11.3343 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 20.248%
```

fp32:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V05 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 19.8213 ms
Effective Bandwidth: 10.1571 GB/s
Effective TFLOPS: 6.93391 TFLOPS
Custom GEMM Kernel Performance
Latency: 35.7192 ms
Effective Bandwidth: 5.63637 GB/s
Effective TFLOPS: 3.84776 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 55.492%
```

From the above results we can see that the performance has signifacntly improved, in FP16 we have 11.3343 TFLOPS, in FP32 we have 3.84776 TFLOPS. Also the latency is significantly reduced in the 2D Block Tiling, 2D Thread Tiling, and Matrix Transpose with Vectorized Memory Access implementation. This is because we are transposing the matrix A and B in the shared memory to make the memory access more coalesced, and also we are using the vectorized memory access to read the values from the shared memory and to write the values to the global memory. This is a trade-off between the performance and the memory bandwidth. This implementation is more efficient than the previous implementation, and it is also more flexible since it can be used for any matrix size.

### 06: Implementation with 2D Block Tiling, 2D Warp Tiling, 2D Thread Tiling, and Matrix Transpose with Vectorized Memory Access
- Static assert things like block_tile, thread tile, no of warps, etc —> this is done to match the hardware capabilities
- Cache A_vals & B_vals in the register
- Define thread_linear_idx, warp_linear_idx, warp_row_idx, warp_col_idx, thread_linear_idx_in_warp, thread_linear_row_idx_in_warp, thread_linear_col_idx_in_warp, num_thread_block_tiles.
- Accumulate the results in registers for each thread.
- Use vectorized memory access.
- Instead of 8 individual 4-byte loads (8 transactions), load as 2 int4 chunks (2 transactions), reducing memory overhead and bandwidth underutilization.
- Tiling over k chunks, since we can’t fit all of k into shared mem at once, we iterate over k chunks of BLOCK_TILE_SIZE_K
    - Load block_tile_size_k —> wide slice of A & B into shared memory
    - All threads in a block compute their piece using the slice
    - Accumulate results in C_thread_results
    - Move to the next K-slice and repeat
- Compute loop
    - for each (thread_tile_repeat_row, thread_tile_repeat_col) pair:
        - grab one A value and one B value
        - multiply and accumulate into C
- Then we do the same nesting as compute loop, but now we iterate over vectorized chunk in X
    - Map from local coords to global matrix coords
    - boundary check
    - Vectorized read from DRAM
    - for each chunk of 4 values
        - Scale compute results by alpha
        - Add beta x old C value
        - Store back in int4 chunk
    - vectorized write to C


fp16:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V06 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 2.45965 ms
Effective Bandwidth: 81.8518 GB/s
Effective TFLOPS: 55.8775 TFLOPS
Custom GEMM Kernel Performance
Latency: 12.2127 ms
Effective Bandwidth: 16.485 GB/s
Effective TFLOPS: 11.2537 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 20.14%
```

fp32:
```bash
Device Name: Tesla T4
Memory Size: 14.5806 GB
Peak Bandwitdh: 320.064 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V06 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 18.2352 ms
Effective Bandwidth: 11.0405 GB/s
Effective TFLOPS: 7.53701 TFLOPS
Custom GEMM Kernel Performance
Latency: 30.7568 ms
Effective Bandwidth: 6.54577 GB/s
Effective TFLOPS: 4.46858 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 59.2884%
```

From the above results we can see that the performance has increased significantly in both FP32. There's a slight decrease in performance in FP16, but it is still a significant improvement over the previous implementation. Why the improvement? Because we are using the vectorized memory access to read the values from the shared memory and to write the values to the global memory. This is a trade-off between the performance and the memory bandwidth. This implementation is more efficient than the previous implementation, and it is also more flexible since it can be used for any matrix size.