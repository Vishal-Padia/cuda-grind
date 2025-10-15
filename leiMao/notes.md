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
