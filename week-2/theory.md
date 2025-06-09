### Some basic concepts

#### `program_id(axis)`

The returns the identifier of the current program instance along the specified axis of the launch grid. The triton kernels are launched over a multi-dimensional grid (typically 1D, 2D, or 3D) where each program instance is responsible for processing a tile (or block) of data.

The axis parameter (0, 1 or 2) specifies which dimension of the gird you want the ID for, analogous to `blockIdx.x`, `blockIdx.y` and `blockIdx.z` in CUDA.

For Example: In a 2D grid, `program_id(0)` gives the row index, and `program_id(1)` gives the column index of the current program instance.

Basically this is used to compute which subset of the input data a given program instance should process, enabling parallelism by dividing the workload across the GPU.

### 2D Block Decomposition

This refers to dividing the overall computation (eg a matrix) into smaller 2D tiles or blocks, with each program instance (kernel launch) responsible for a specific tile. This is common in matrix operations (like GEMM, convolution) where the data is naturally two-dimensional.

Why 2D? Many problems (matrices, images) are inherently 2D, and decomposing them into blocks allows for efficient parallel processing and better utilization of GPU resources.

How in Triton? We can launch our kernel over a 2D grid, and each program instance uses its (row, col) indicies from `program_id(0)` and `program_id(1)` to determine which tile of the matrix should it process.

### Row/Column Indexing

Basically we can use the row and column indices to access the elements of the matrix.

### Cache-friendly Memory Access

This means arranging our memory accesses so that they are contiguous or coalesced, allowing the GPU to fetch data in large, aligned blocks rather than scattered, inefficient accesses.

Why it matters: GPUs have high memory bandwidth, but only if accesses are coalesced, Strided or random accesses can lead to cache misses and underutilized bandwidth.

In Triton: When designing our block-decomposition and indexing, we want each program instance to access memory in a way that consecutive threads (or program instances) access consecutive memory locations. This is often achieved by:
    - Assigning each program instance a contiguous block of rows or columns.
    - Ensuring accesses within a tile are along the fastest-changing dimension (row-major or column-major, depending on data layout)

Example: In a matrix, if each program instance processes a tile of consecutives rows, and within each tile accesses elements row-wise, memory accesses are coalesced maximizing cache efficiency.

### Summary

- Triton Kernels are launched over a multi-dimesional grid
- Each kernel instance identifies its tile via `program_id(axis)`
- 2D block decomposition assigns each instance a unique tile of the data
- Row/column indexing computes the start of the tile for each instance
- Cache-friendly access ensures each instance reads/writes data in a way that aligns with GPU memory architecture, maximizing throughput and minimizing stalls
