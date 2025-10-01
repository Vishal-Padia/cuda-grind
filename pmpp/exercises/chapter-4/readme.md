### Chapter 4: Exercises

4.1 If a CUDA device’s SM can take up to 1,536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?

  - A) 128 threads per block
  - B) 256 threads per block
  - C) 512 threads per block
  - D) 1024 threads per block

> Here max number of threads is $T_{max} = 1536$ and max number of blocks is $B_{max} = 4$.
> Number of blocks that can fit is $N_{blocks} = \min(B_{max}, \frac{T_{max}}{t_{b}}) = \min(4, 3) = 3$ where $t_{b}$ is the number of threads per block.
> We can now substitute $t_{b}$ with the values from the options:
> $\min(4, \frac{1536}{128}) = \min(4, 12) = 4 * 128 = 512$ --> So this doesn't work out for us.
> $\min(4, \frac{1536}{256}) = \min(4, 6) = 4 * 256 = 1024$ --> So this doesn't work out for us.
> $\min(4, \frac{1536}{512}) = \min(4, 3) = 3 * 512 = 1536$ --> This is the correct answer.
> $\min(4, \frac{1536}{1024}) = \min(4, 1) = 1 * 1024 = 4096$ --> So this doesn't work out for us. (We round here)

> The correct answer is C - 512 threads per block

4.2 For a vector addition, assume that the vector length is 2,000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

  - A) 2,000
  - B) 2,024
  - C) 2,048
  - D) 2,096

> The correct answer is C - 2048, because the block size is 512 threads per block and the vector length is 2,000 elements, we will have 4 blocks of 512 threads, we would rather have more threads than less.

4.3 For the previous question, how many warps do you expect to have divergence due to the boundary check on the vector length?

  - A) 1
  - B) 2
  - C) 3
  - D) 6

> The correct answer here is A - 1, because the warp size is 32 threads (standard for cuda), and there are 48 unused threads in the last block. Since the offset number is 32, thread index 1984 - 2015 contains both used and unused threads, and post that ie after 2016 - 20248 there are no used threads, so there is no divergence there. Only the warp containing used and unused threads will have divergence. Therefore the answer is 1.

4.4  You need to write a kernel that operates on an image of size `400x900` pixels. You would like to assign one thread to each pixel. You would like your thread blocks to be square and to use the maximum number of threads per block possible on the device (your device has compute capability 3.0). How would you select the grid dimensions and block dimensions of your kernel?

> For 3.0 we have a max of 1024 threads per block. so NxN where N^2 <=1024, so 32 is a good start. So our block dimensions are 32x32. We can now calculate the grid dimensions as follows:
>
```
  Grid dimensions = (image width / block width) x (image height / block height)
  Grid dimensions = (400 / 32) x (900 / 32)
  Grid dimensions = 12.5 x 28.125
  Since we cannot have a fraction of a block, we round up to the nearest integer:
  Grid dimensions = 13 x 29
  ```

4.5 For the previous question, how many idle threads do you expect to have?

> (13 * 29) * (32 * 32) = 3,86,048
> so idle threads will be total_threads - total_threads_required = 3,86,048 - 3,60,000 (which is the total number of pixels) = 26,048.

4.6 Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9, and spend the rest of their time waiting for the barrier. What percentage of the threads' summed-up execution times is spent waiting for the barrier?

> the max is 3.0, we subtract each from the max, then add them and then take the percentage of that.
> Percentage_waiting = ((sum of waiting times) / (sum of all working times) + (sum of all waiting times)) * 100
> ((3.0 - 2.0) + (3.0 - 2.3) + (3.0 - 3.0) + (3.0 - 2.8) + (3.0 - 2.4) + (3.0 - 1.9) + (3.0 - 2.6) + (3.0 - 2.9)) / ((2.0 + 2.3 + 3.0 + 2.8 + 2.4 + 1.9 + 2.6 + 2.9) + (3.0 - 2.0) + (3.0 - 2.3) + (3.0 - 3.0) + (3.0 - 2.8) + (3.0 - 2.4) + (3.0 - 1.9) + (3.0 - 2.6) + (3.0 - 2.9)) * 100 = 17.083%

4.7 Indicate which of the following assignments per multiprocessor is possible. In the case where it is not possible, indicate the limiting factor(s).

  - A) 8 blocks with 128 threads each on a device with compute capability 1.0
  - B) 8 blocks with 128 threads each on a device with compute capability 1.2
  - C) 8 blocks with 128 threads each on a device with compute capability 3.0
  - D) 16 blocks with 64 threads each on a device with compute capability 1.0
  - E) 16 blocks with 64 threads each on a device with compute capability 1.2
  - F) 16 blocks with 64 threads each on a device with compute capability 3.0

> Option B and E are not possible because they exceed the maximum number of threads per block and blocks per multiprocessor respectively.

4.8 A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the __syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

> No I don't think this is a good idea. Because we should Never rely on the current hardware's warp size or scheduling for correctness. Always use `__syncthreads()` when you need barrier synchronization within a block, regardless of block size.

4.9 A student mentioned that he was able to multiply two 1,024x1,024 matrices using a tiled matrix multiplication code with 32x32 thread blocks. He is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. He further mentioned that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

> The student claims to use 32x32 = 1024 threads per block, which exceeds the hardware capability that they have mentioned which is 512 threads per block. This is not possible with the hardware they are using.

4.10 The following kernel is executed on a large matrix, which is tiled into submatrices. To manipulate tiles, a new CUDA programmer has written the following device kernel, which will transpose each tile in the matrix. The tiles are of size `BLOCK_WIDTH` by `BLOCK_WIDTH`, and each of the dimensions of matrix A is known to be a multiple of `BLOCK_WIDTH`. The kernel invocation and code are shown below. `BLOCK_WIDTH` is known at compile time, but could be set anywhere from 1 to 20.
```cpp
dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x,A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);
__global__ void BlockTranspose(float A_elements, int A_width, int A_height)
{
  __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
  int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;
  blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
  A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

  - A) Out of the possible range of values for `BLOCK_SIZE`, for what values of `BLOCK_SIZE` will this kernel function correctly when executing on the device?
  > It will execute for all possible values if the data in shared memory is fully written before any thread reads from it.

  - B) If the code does not execute correctly for all `BLOCK_SIZE` values, suggest a fix to the code to make it work for all `BLOCK_SIZE` values.
  > To ensure that the data in shared memory is fully written before any thread reads from it, we can use the `__syncthreads()` function to synchronize all threads in the block before reading from shared memory. This ensures that all threads have completed their writes to shared memory before any thread reads from it.
  ```cpp
  dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH);
  dim3 gridDim(A_width/blockDim.x,A_height/blockDim.y);
  BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);
  __global__ void BlockTranspose(float A_elements, int A_width, int A_height)
  {
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;
    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    __syncthreads();
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
  }
  ```
