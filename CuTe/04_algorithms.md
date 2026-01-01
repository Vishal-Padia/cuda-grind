# Algorithms

### Whate is a CuTe Algorithm?
> A compile-time transformation that maps execution indices (threads, warps, blocks) to tensor subviews.

Important:
- Algorithms do no touch memory
- They do not compute offsets manually
- They produce tensors (views)

Think of algorithms as:
> "Structured ways to slice tensors parallel execution"

### The core problem algorithms solve

In a CUDA kernel, we always face this:
- Threads exists in a grid/block
- Data exists in a tensor
- You must map
    ```
    thread -> element(s)
    ```

Naive CUDA:
```
int i = blockIdx.x * blockDim.x + threadIdx.x;
ptr[i] = ...;
```

CuTe replaces this with:
```
auto t = local_partition(tensor, thread_layout, threadIdx.x);
```

No index math, No conditionals, Layout-driven.

### `local_partition` - The most important algorithms

What it does?
> `local_partition` slices a tensor so each thread sees only its assigned portion.

We give it:
- a tensor
- a thread layout
- a thread id

It gives you:
- a subtensor for that thread

Minimal Example (1D):

Tensor
```
auto tensor = make_tensor(ptr, make_layout(make_shape(128), make_stride(1)));
```

Threads, Assume 128 threads, one element each
```
auto thread_layout = make_layout(make_shape(128), make_stride(1));
```

Partition
```
auto t = local_partition(tensor, thread_layout, threadIdx.x);
```
Now:
- `t` is a scalar tensor
- Thread `k` sees element `tensor(k)`

### Why is this better than index math
Because:
- The layout guarantees correctness
- Changing mapping = changing layout
- Kernel code stays the same

This is how CUTLASS supports:
- different tile sizes
- different warp shapes
- different memory layouts

### 2D Example - Threads x Matrix

Let's say:
- Matrix: `(16, 16)`
- Threads: `(16, 16)` (one thread per element)

Tensor:
```
auto A = make_tensor(ptr, make_layout(make_shape(16, 16), make_stride(16, 1)));
```

Thread Layout:
```
auto T = make_layout(make_shape(16, 16), make_stride(16, 1));
```

Partition
```
auto a = local_partition(A, T, make_coord(threadIdx.y, threadIdx.x));
```
Now:
- each thread sees one scalar
- `a()` access the correct element
- no index math anywhere

### Tiling with Algorithms (Real use case)

Global Tensor
```
auto A = make_tensor(ptr,
    make_layout(make_shape(128, 128), make_stride(128, 1))
);
```

Tile Layout (per block)
```
auto block_tile = make_layout(make_shape(16, 16), make_stride(16, 1));
```

Partition by block
```
auto A_block = local_partition(A, block_tile, make_coord(blockIdx.y, blockIdx.x));
```

Now:
- `A_block` is a 16x16 subtensor
- Each block sees a different tile
- No pointer arithmetic written by you

### Algorithms Compose
```
global tensor
  ↓ local_partition (block)
block tile
  ↓ local_partition (warp)
warp tile
  ↓ local_partition (thread)
scalar or fragment
```

Each step:
- returns a tensor
- preserves layout semantics
- composes cleanly

This is how FlashAttention and CUTLASS kernels are structured

> Algorithms decide "who sees what", layouts decide "where it lives"

If you keep this separation, CuTe stays clean and powerful.

### Exercise 1
>What problem do CuTe algorithms solve that layouts alone cannot?

Answer: It solves the problem of mapping threads to elements in a tensor.

### Exercise 2
Suppose:
- Tensor shape: `(64, 64)`
- Block tile: `(16, 16)`
- Grid: `(4, 4)`

Answer:
1. What tile does block `(2, 1)` get?
> The `(2, 1)` tile in the `(4, 4)` tile grid.

2. What logical indices does that tile cover?
> Rows [32....47] and Columns [16....31]

### Exercise 3
> CuTe algorithms allow me to write kernels that are __ with respect to layout and __ with respect to execution

Answer:
CuTe algorithms allow me to write kernels that are *flexible* with respect to layout and *correct* with respect to execution.