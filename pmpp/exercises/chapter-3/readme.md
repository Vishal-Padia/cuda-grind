### Chapter 3: Exercises

3.1. A matrix addition takes two input matrices B and C and produces one output matrix A. Each element of the output matrix A is the sum of the corresponding elements of the input matrices B and C, that is, $\(A_{ij} = B_{ij} + C_{ij}\)$. For simplicity, handle only square matrices with single-precision floating-point numbers.

a) Write a matrix addition kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the first input matrix, pointer to the second input matrix, and the number of elements in each dimension.

> DONE. Checkout [matrix_addition_kernel.cu](matrix_addition_kernel.cu)

b) Write the host stub function for memory allocation, data transfer, kernel launch (leave execution config open), and memory free.

> DONE. Checkout [matrix_addition_kernel.cu](matrix_addition_kernel.cu)

c) Write a kernel where each thread produces one output matrix element. Complete the execution config for this design.

> DONE. Checkout [matrix_addition_kernel.cu](matrix_addition_kernel.cu)

d) Write a kernel where each thread produces one output matrix row. Complete the execution config for this design.

> DONE. Checkout [matrix_addition_kernel_row.cu](matrix_addition_kernel_row.cu)

e) Write a kernel where each thread produces one output matrix column. Complete the execution config for this design.

> DONE. Checkout [matrix_addition_kernel_column.cu](matrix_addition_kernel_column.cu)

f) Analyze the pros and cons of each preceding kernel design.

| Design | Pros | Cons |
|--------|------|------|
| One thread per element | Max parallelism, simple, balanced workload | High launch overhead, resource contention possible |
| One thread per row | Fewer threads, better compute granularity | Less parallelism, possible register pressure |
| One thread per column | Similar to row-wise, potential column reuse | Uncoalesced accesses, inefficient memory bandwidth

<br>

3.2. A matrix-vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of A is the dot product of one row of B and C, i.e., $\(A_i = \sum_j B_{ij} \cdot C_j\)$. Again, only consider square matrices with single-precision floats. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension.

> DONE. Checkout [matrix_vector_multiplication_kernel.cu](matrix_vector_multiplication_kernel.cu)

3.3. A new summer intern complains about CUDA’s tediousness since he has to declare many functions he plans to execute on both host and device twice (once as a host function and once as a device function). What is your response?

> The intern can use the `__host__` and `__device__` keywords to declare the function once and have the compiler generate the host and device versions automatically.

3.4. Complete Parts 1 and 2 of the function in Figure 3.6.

> DONE.

3.5. If one thread calculates one output element of a vector addition, what is the correct expression for mapping thread/block indices to data index?

    - A. $\(i = \text{threadIdx.x} + \text{threadIdx.y}\)$

    - B. $\(i = \text{blockIdx.x} + \text{threadIdx.x}\)$

    - C. $\(i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}\)$

    - D. $\(i = \text{blockIdx.x} \text{threadIdx.x}\)$

> The correct answer here is C. $\(i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}\)$

3.6. Each thread should calculate two adjacent elements of a vector addition. What is the correct expression for the starting index?

    - A. $\(i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x} \times 2\)$

    - B. $\(i = \text{blockIdx.x} \text{threadIdx.x} \times 2\)$

    - C. $\(i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x} \times 2\)$

    - D. $\(i = (\text{blockIdx.x} \times \text{blockDim.x} \times 2) + \text{threadIdx.x}\)$

> The correct answer here is D. $\(i = (\text{blockIdx.x} \times \text{blockDim.x} \times 2) + \text{threadIdx.x}\)$

3.7. For vector addition, if vector length is 2000, each thread calculates one output element, and block size is 512 threads, how many threads are in the grid?

    - A. 2000

    - B. 2024

    - C. 2048

    - D. 2096

> To calculate the number of threads in the grid, we need to divide vector length by the block size ie 2000/512 = 3.90625 rouding this up to 4. So 4 * 512 = 2,048, which is the number of threads in the grid. Therefore the correct answer is C 2048.

$$ \text{totalThreads} = \text{blocksPerGrid} \times \text{blockSize} $$
$$ \text{blocksPerGrid} = \lceil \frac{\text{vectorLength}}{\text{blockSize}} \rceil $$
