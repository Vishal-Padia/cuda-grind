### Chapter 5 Exercises

5.1. Consider the matrix addition in Exercise 3.1. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: analyze the elements accessed by each thread and see if there is any commonality between threads.

> Yes, we can use shared memory to reduce the global memory bandwidth consumption. We can "carpool" threads who are trying to access the same data. This means that multiple threads can access the same data from shared memory instead of accessing it from global memory multiple times. By doing so, we can reduce the number of global memory accesses and thus reduce the global memory bandwidth consumption.

5.2. Draw the equivalent of Figure 5.6 for an $$8 \times 8$$ matrix multiplication with $$2 \times 2$$ tiling and $$4 \times 4$$ tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.

> 2x2 tiling for 8x8 matrix

| Thread      | Access 1          | Access 2          | Access 3          | Access 4          | Access 5          | Access 6          | Access 7          | Access 8          |
|-------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| thread_0,0  | M_0,0 * N_0,0     | M_0,1 * N_1,0     | M_0,2 * N_2,0     | M_0,3 * N_3,0     | M_0,4 * N_4,0     | M_0,5 * N_5,0     | M_0,6 * N_6,0     | M_0,7 * N_7,0     |
| thread_0,1  | M_0,0 * N_0,1     | M_0,1 * N_1,1     | M_0,2 * N_2,1     | M_0,3 * N_3,1     | M_0,4 * N_4,1     | M_0,5 * N_5,1     | M_0,6 * N_6,1     | M_0,7 * N_7,1     |
| thread_1,0  | M_1,0 * N_0,0     | M_1,1 * N_1,0     | M_1,2 * N_2,0     | M_1,3 * N_3,0     | M_1,4 * N_4,0     | M_1,5 * N_5,0     | M_1,6 * N_6,0     | M_1,7 * N_7,0     |
| thread_1,1  | M_1,0 * N_0,1     | M_1,1 * N_1,1     | M_1,2 * N_2,1     | M_1,3 * N_3,1     | M_1,4 * N_4,1     | M_1,5 * N_5,1     | M_1,6 * N_6,1     | M_1,7 * N_7,1     |
| thread_2,0  | M_2,0 * N_0,0     | M_2,1 * N_1,0     | M_2,2 * N_2,0     | M_2,3 * N_3,0     | M_2,4 * N_4,0     | M_2,5 * N_5,0     | M_2,6 * N_6,0     | M_2,7 * N_7,0     |
| thread_2,1  | M_2,0 * N_0,1     | M_2,1 * N_1,1     | M_2,2 * N_2,1     | M_2,3 * N_3,1     | M_2,4 * N_4,1     | M_2,5 * N_5,1     | M_2,6 * N_6,1     | M_2,7 * N_7,1     |
| thread_3,0  | M_3,0 * N_0,0     | M_3,1 * N_1,0     | M_3,2 * N_2,0     | M_3,3 * N_3,0     | M_3,4 * N_4,0     | M_3,5 * N_5,0     | M_3,6 * N_6,0     | M_3,7 * N_7,0     |
| thread_3,1  | M_3,0 * N_0,1     | M_3,1 * N_1,1     | M_3,2 * N_2,1     | M_3,3 * N_3,1     | M_3,4 * N_4,1     | M_3,5 * N_5,1     | M_3,6 * N_6,1     | M_3,7 * N_7,1     |

> 4x4 tiling for 8x8 matrix

| Thread      | Access 1          | Access 2          | Access 3          | Access 4          | Access 5          | Access 6          | Access 7          | Access 8          |
|-------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| thread_0,0  | M_0,0 * N_0,0     | M_0,1 * N_1,0     | M_0,2 * N_2,0     | M_0,3 * N_3,0     | M_0,4 * N_4,0     | M_0,5 * N_5,0     | M_0,6 * N_6,0     | M_0,7 * N_7,0     |
| thread_0,1  | M_0,0 * N_0,1     | M_0,1 * N_1,1     | M_0,2 * N_2,1     | M_0,3 * N_3,1     | M_0,4 * N_4,1     | M_0,5 * N_5,1     | M_0,6 * N_6,1     | M_0,7 * N_7,1     |
| thread_0,2  | M_0,0 * N_0,2     | M_0,1 * N_1,2     | M_0,2 * N_2,2     | M_0,3 * N_3,2     | M_0,4 * N_4,2     | M_0,5 * N_5,2     | M_0,6 * N_6,2     | M_0,7 * N_7,2     |
| thread_0,3  | M_0,0 * N_0,3     | M_0,1 * N_1,3     | M_0,2 * N_2,3     | M_0,3 * N_3,3     | M_0,4 * N_4,3     | M_0,5 * N_5,3     | M_0,6 * N_6,3     | M_0,7 * N_7,3     |
| thread_1,0  | M_1,0 * N_0,0     | M_1,1 * N_1,0     | M_1,2 * N_2,0     | M_1,3 * N_3,0     | M_1,4 * N_4,0     | M_1,5 * N_5,0     | M_1,6 * N_6,0     | M_1,7 * N_7,0     |
| thread_1,1  | M_1,0 * N_0,1     | M_1,1 * N_1,1     | M_1,2 * N_2,1     | M_1,3 * N_3,1     | M_1,4 * N_4,1     | M_1,5 * N_5,1     | M_1,6 * N_6,1     | M_1,7 * N_7,1     |
| thread_1,2  | M_1,0 * N_0,2     | M_1,1 * N_1,2     | M_1,2 * N_2,2     | M_1,3 * N_3,2     | M_1,4 * N_4,2     | M_1,5 * N_5,2     | M_1,6 * N_6,2     | M_1,7 * N_7,2     |
| thread_1,3  | M_1,0 * N_0,3     | M_1,1 * N_1,3     | M_1,2 * N_2,3     | M_1,3 * N_3,3     | M_1,4 * N_4,3     | M_1,5 * N_5,3     | M_1,6 * N_6,3     | M_1,7 * N_7,3     |
| thread_2,0  | M_2,0 * N_0,0     | M_2,1 * N_1,0     | M_2,2 * N_2,0     | M_2,3 * N_3,0     | M_2,4 * N_4,0     | M_2,5 * N_5,0     | M_2,6 * N_6,0     | M_2,7 * N_7,0     |
| thread_2,1  | M_2,0 * N_0,1     | M_2,1 * N_1,1     | M_2,2 * N_2,1     | M_2,3 * N_3,1     | M_2,4 * N_4,1     | M_2,5 * N_5,1     | M_2,6 * N_6,1     | M_2,7 * N_7,1     |
| thread_2,2  | M_2,0 * N_0,2     | M_2,1 * N_1,2     | M_2,2 * N_2,2     | M_2,3 * N_3,2     | M_2,4 * N_4,2     | M_2,5 * N_5,2     | M_2,6 * N_6,2     | M_2,7 * N_7,2     |
| thread_2,3  | M_2,0 * N_0,3     | M_2,1 * N_1,3     | M_2,2 * N_2,3     | M_2,3 * N_3,3     | M_2,4 * N_4,3     | M_2,5 * N_5,3     | M_2,6 * N_6,3     | M_2,7 * N_7,3     |
| thread_3,0  | M_3,0 * N_0,0     | M_3,1 * N_1,0     | M_3,2 * N_2,0     | M_3,3 * N_3,0     | M_3,4 * N_4,0     | M_3,5 * N_5,0     | M_3,6 * N_6,0     | M_3,7 * N_7,0     |
| thread_3,1  | M_3,0 * N_0,1     | M_3,1 * N_1,1     | M_3,2 * N_2,1     | M_3,3 * N_3,1     | M_3,4 * N_4,1     | M_3,5 * N_5,1     | M_3,6 * N_6,1     | M_3,7 * N_7,1     |
| thread_3,2  | M_3,0 * N_0,2     | M_3,1 * N_1,2     | M_3,2 * N_2,2     | M_3,3 * N_3,2     | M_3,4 * N_4,2     | M_3,5 * N_5,2     | M_3,6 * N_6,2     | M_3,7 * N_7,2     |
| thread_3,3  | M_3,0 * N_0,3     | M_3,1 * N_1,3     | M_3,2 * N_2,3     | M_3,3 * N_3,3     | M_3,4 * N_4,3     | M_3,5 * N_5,3     | M_3,6 * N_6,3     | M_3,7 * N_7,3     |


5.3. What type of incorrect execution behavior can happen if one forgets to use `syncthreads` in the kernel of Figure 5.12?

> If we don't use `syncthreads`, we can get a race condition where threads are reading and writing to the same memory location, leading to unpredictable results.

5.4. Assuming capacity was not an issue for registers or shared memory, give one case that it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.

> whenever data from global memory needs to be reused within a block across multiple threads, shared memory is preferable to registers—even if registers could technically store all the needed values.

5.5. For the tiled matrix-matrix multiplication kernel, if we use a $$32 \times 32$$ tile, what is the reduction of memory bandwidth usage for input matrices $$M$$ and $$N$$?
  - a. $$1/8$$ of the original usage
  - b. $$1/16$$ of the original usage
  - c. $$1/32$$ of the original usage
  - d. $$1/64$$ of the original usage

> C. $$1/32$$ of the original usage

5.6. Assume that a kernel is launched with 1,000 thread blocks each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?
  - a. 1
  - b. 1,000
  - c. 512
  - d. 512,000

> D. 512,000

5.7. In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?
  - a. 1
  - b. 1,000
  - c. 512
  - d. 51,200

> C. 512

5.8. Explain the difference between shared memory and L1 cache.

> Shared memory is a region of memory that is shared among all threads in a block. L1 cache is a small, fast memory that is shared among all cores in a processor. The data in L1 cache is managed the hardware itself unlike shared memory which is managed by the programmer.

5.9. Consider performing a matrix multiplication of two input matrices with dimensions $$N \times N$$. How many times is each element in the input matrices requested from global memory when
  - a. There is no tiling?
  - b. Tiles of size $$T \times T$$ are used?

> A. When there is no tiling, each element in the input matrices is requested from global memory once.

> B. When tiles of size $$T \times T$$ are used, each element in the input matrices is requested from global memory $$\frac{N}{T}$$ times.

5.10. A kernel performs 36 floating-point operations and 7 32-bit word global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute- or memory-bound.
  - a. Peak FLOPS = 200 GFLOPS, peak memory bandwidth = 100 GB/s.
  - b. Peak FLOPS = 300 GFLOPS, peak memory bandwidth = 250 GB/s.

> Arithmetic intensity of the kernel : $$\frac{36}{7}$$ = 5.12 Flops per memory.

> For A: $$\frac{200}{100}$$ = 2 flops per memory, so 5.12 > 2, hence it's compute bound.

> For B: $$\frac{300}{250}$$ = 1.2 flops per memory, so 5.12 > 1.2, hence it's compute bound.

5.11. Indicate which of the following assignments per streaming multiprocessor is possible. In the case where it is not possible, indicate the limiting factors.
  - a. 4 blocks with 128 threads each and 32 B shared memory per thread on a device with compute capability 1.0.
  - b. 8 blocks with 128 threads each and 16 B shared memory per thread on a device with compute capability 1.0.
  - c. 16 blocks with 32 threads each and 64 B shared memory per thread on a device with compute capability 1.0.
  - d. 2 blocks with 512 threads each and 32 B shared memory per thread on a device with compute capability 1.2.
  - e. 4 blocks with 256 threads each and 16 B shared memory per thread on a device with compute capability 1.2.
  - f. 8 blocks with 256 threads each and 8 B shared memory per thread on a device with compute capability 1.2.

> We can calculate this by (threads * shared memory per thread) * number of blocks.

> For A: (128 * 32) * 4 = 16384 B, Max capacity for 1.0 is 16kb --> so its possible

> For B: (128 * 16) * 8 = 16384 B, Max capacity for 1.0 is 16kb --> so its possible

> For C: (32 * 64) * 16 = 32768 B, Max capacity for 1.0 is 16kb --> so its not possible

> For D: (512 * 32) * 2 = 32768 B, Max capacity for 1.2 is 16kb --> so its not possible

> For E: (256 * 16) * 4 = 16384 B, Max capacity for 1.2 is 16kb --> so its possible

> For F: (256 * 8) * 8 = 16384 B, Max capacity for 1.2 is 16kb --> so its possible

> So if the value is within the limit of the device, it is possible otherwise it's not possible.
