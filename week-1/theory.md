# GPU Architecture Basics

### Streaming Multiprocessors (SMs)

- The core building block of an NVIDIA GPU is the Streaming Multiprocessor (SM)
- Each SM contains many simple processing cores, on-chip shared memory, registers and control logic for managing thousands of lightweight threads
- When we launch a kernel, CUDA divides the work into ___blocks___ of threads. Each block is scheduled onto an SM, where it's threads execute concurrently
- Multiple blocks can be active on a single SM, depending on resource usage (registers, shared memory, etc.)

### Threads, Wraps and Blocks
- Threads are the smallest unit of execution. A ___wrap___ is a group of 32 threads that execute instructions in lockstep (SIMT: single instruction, multiple threads)
- Threads are grouped into ___blocks___. All threads in a block can share data via ___shared memory___ and can synchronize with each other
- Blocks are grouped into a ___grid___ for kernel launch. Blocks are independent and can be scheduled in any order across SMs

### Memory Hierarchy
- Each thread has private ___local memory___ (registers and local variables)
- Each block has ___shared memory___, fast and accessible by all threads in the block
- All threads have access to ___global memory___ (device memory), which is large but slower
- There are also constant, texture and surface memories, optimized for specific access patterns

### `__global__` vs `__device__`
- `__global__` function are kernels: callable from the host (CPU) and executed on the device (GPU) by many threads in parallel
- `__device__` function are callable only from other device or global functions, and execute on the device (GPU) as a single thread per call

### Thread and Block Indexing
- Each thread gets unique indices to identify its position within its block and grid
- Use `threadIdx.{x, y, z}` for a thread's index within its block
- Use `blockIdx.{x, y, z}` for a block's index within the grid
- Use `blockDim.{x, y, z}` for the size of each block
- Use `gridDim.{x, y, z}` for the size of grid

This allows us to compute a unique global index for each thread, so you can map threads to data elements efficiently

### Host <--> Device Memory Management
- The host (CPU) and deivce (GPU) each have their own memory spaces
- Data must be explicitly copied between host and device using CUDA API calls (eg `cudaMemcpy`)
- Typical workflow:
    1. Allocate memory on the device
    2. Copy data from host to device
    3. Launch Kernel(s)
    4. Copy results from device to host
    5. Free device memory

Efficient memory management and minimizing transfers is crucial for performance as PCIe transfers are much slower than on-device memory access

### Summary:
- SMs: Parallel execution units, each running many threads
- Threads/Wraps/Blocks: Hierarchical organization for massive parallelism
- Memory Hierarchy: Local, shared, global, and specialized memories
- `__global__` vs `__device__`: Kernel vs device function usage
- Thread/Block Indexing: Mechanism for distributing work/data
- Host/Device Memory Management: Explicit data movement between CPU and GPU
