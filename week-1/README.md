Here are the things I'll be doing in this week:

**Goal**: Master the basics of Triton JIT, program_id, and memory ops.

**Concepts**:

- `@triton.jit`, `program_id()`
- `tl.load`, `tl.store`, broadcasting
- Memory layouts (strided, contiguous), boundary masking

**Projects**:

1. `vector_add(A, B) → C`
2. `vector_mul(A, B) → C` (test mixed strides)
3. `vector_relu(A) → C` (elementwise activation)

**Benchmark**:

- Compare with NumPy and Torch on CPU/GPU
- Use Triton’s `triton.testing.perf_report`

**Extensions**:

- Autotune block size for vector ops
- Visualize memory pattern using `nsys` / `nvprof`