#include <iostream>
#include <math.h>

// kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{   
    // int index = threadIdx.x; // get the thread index
    // int stride = blockDim.x; // get the number of threads in the block

    int index = blockIdx.x * blockDim.x + threadIdx.x; // get the global thread index
    int stride = blockDim.x * gridDim.x; // get the total number of threads in the grid

    for (int i = index; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
    }
}

int main(void)
{
    int N = 1<<20;
    float *x, *y;

    // allocate unified memory - accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run kernel on 1M elements on the GPU
    // add<<<1, 256>>>(N, x, y); // 1 block of 256 threads

    // calcuating the blocks and threads
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // prefetch the x and y arrays to the GPU
    cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
    cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

    // launch the kernel
    add<<<numBlocks, blockSize>>>(N, x, y);

    // wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // check for errors (all values should be 3.0f)
    float maxError = 0.0f;

    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }

    std::cout << "Max Error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
