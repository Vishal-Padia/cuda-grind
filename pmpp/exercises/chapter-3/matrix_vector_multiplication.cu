#include <iostream>
#include <cuda_runtime.h>

// matrix multiplication kernel
__global__ void Matrix_Vector_Multiplication(float* A, const float* B, const float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += B[row * N + j] * C[j];
        }
        A[row] = sum;
    }
}

int main() {
    int N = 1024;
    size_t bytes = N * N * sizeof(float);
    size_t bytes_vector = N * sizeof(float);

    // allocate memory
    float *h_A = (float*)malloc(bytes_vector);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_vector);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // initialize matrix
    for (int i = 0; i < N * N; ++i) {
        h_B[i] = static_cast<float>(i);
    }

    // initialize vector
    for (int i = 0; i < N; ++i) {
        h_C[i] = static_cast<float>(i);
    }

    // copy from host to device
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    Matrix_Vector_Multiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // copy data from device to host
    cudaMemcpy(h_A, d_A, bytes_vector, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        std::cout << h_A[i] << "\n";
    }

    // free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}
