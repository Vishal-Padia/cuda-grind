#include <iostream>
#include <cuda_runtime.h>

// produces one output matrix column
__global__ void MatrixAdditionColumn(const float* B, const float* C, float* A, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        for (int row = 0; row < N; ++row) {
            int idx = row * N + col; // column-major order
            A[idx] = B[idx] + C[idx];
        }
    }
}


int main() {
    int N = 1024;
    size_t bytes = N * N * sizeof(float);

    // allocate memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_B[i] = static_cast<float>(i);
        h_C[i] = static_cast<float>(i);
    }

    // copy data from host to device
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);

    // Launch kernel for column-wise addition
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    MatrixAdditionColumn<<<blocksPerGrid, threadsPerBlock>>>(d_B, d_C, d_A, N);

    // copy data from device to host
    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        std::cout << h_A[i] << "\n";
    }

    // free up memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

}
