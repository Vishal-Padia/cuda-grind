#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 2

__global__ void MatrixMulKernelTiled(float* d_M, float* d_N, float* d_P, int Width) {
    // declare Mds and Nds - basically the shared memory variables
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // save the threadIdx and blockIdx values into automatic variables
    // and thus into registers for fast access
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // identify the row and col of the d_P element
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    // loop over the d_M and d_N tiles required to compute d_P element
    for (int m = 0; m < Width/TILE_WIDTH; ++m) {
        // collaborative loading of d_M and d_N into shared memory
        Mds[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        // compute the d_P element
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[Row * Width + Col] = Pvalue;
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
    MatrixMulKernelTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, WIDTH);


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