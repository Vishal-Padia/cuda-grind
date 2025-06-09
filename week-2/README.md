Here are the things I'll be doing in this week:

**Goal**: Implement baseline 2D matrix multiplication using block-wise tiling.

**Concepts**:

- `program_id(axis)`
- 2D block decomposition
- Row/column indexing
- Cache-friendly memory access

**Projects**:

1. Naive `matmul(A, B) → C` (no shared memory)
2. Add scalar bias and `ReLU` activation post-matmul

**Benchmark**:

- Validate against `torch.matmul`
- Compare execution time, memory bandwidth

**Extensions**:

- Add support for batch matmul (B, M, N)
- Add broadcasting for bias