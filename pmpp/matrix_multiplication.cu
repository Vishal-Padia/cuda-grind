#include <iostream>
#include <cuda_runtime.h>

__global__ void MatrixMultiplication(float* A, float* B, float* C, int WIDTH) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < WIDTH) && (col < WIDTH)) {
        float sum = 0.0f;
        for (int k = 0; k < WIDTH; ++k) {
            sum += A[row * WIDTH + k] * B[k * WIDTH + col];
        }
        C[row * WIDTH + col] = sum;
    }
}
int main() {
    int WIDTH = 4;
    size_t bytes = WIDTH * WIDTH * sizeof(float);

    // allocate memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // initialize matrices
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            h_A[i * WIDTH + j] = i + j;
            h_B[i * WIDTH + j] = i - j;
        }
    }

    // copy from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // this is the width of each block meaning
    // the WIDTH VAR is the whole chessboard 8x8
    // and this is a small 2x2 tile on the chessboard
    #define BLOCK_WIDTH 16
    int NumBlocks = WIDTH / BLOCK_WIDTH;
    if (WIDTH % BLOCK_WIDTH) NumBlocks++;

    // launch kernel
    dim3 dimGrid(NumBlocks, NumBlocks);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    MatrixMultiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, WIDTH);


    // copy data from device to host and print
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            std::cout << h_C[i * WIDTH + j] << " ";
        }
        std::cout << std::endl;
    }

    // free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}
