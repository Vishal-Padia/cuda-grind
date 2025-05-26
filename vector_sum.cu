#include <iostream>
#include <vector>

#define N 10

__global__ void add(const int *a, const int *b, int *c) {
    int tid = blockIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Use std::vector for host arrays
    std::vector<int> a(N), b(N), c(N);

    int *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;

    // Allocate memory on the GPU
    cudaMalloc(&dev_a, N * sizeof(int));
    cudaMalloc(&dev_b, N * sizeof(int));
    cudaMalloc(&dev_c, N * sizeof(int));

    // Fill the arrays
    for (int i = 0; i < N; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }

    // Copy the arrays to the GPU
    cudaMemcpy(dev_a, a.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    // Copy the result back to the CPU
    cudaMemcpy(c.data(), dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the results using C++ streams
    for (int i = 0; i < N; ++i) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // Free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
