# Layout

### What is a Layout?
A layout answers exactly one question:
> Given a logical index `(i_0, i_1, ..., i_n)`, where does it live in linear memory?

That's it

In other words:
```
Layout : Multi-dimensional index -> Linear offset
```

Why CuTe cares so much about Layouts

On GPUs:
- Memory access pattersn domainates performance
- Strides, transposes, padding, swizzles matter
- You want to change memory traversal without rewriting code

CuTe's solution:
> Make memory mapping a first-class, compile-time object

### Formal Definition
In CuTe, a `Layout` is composed of:
```
Layout = (Shape, Stride)
```
Where:
- Shape: the logical size of each dimensions
- Stride: how much the pointer moves when that dimension increments

Mathematical form

For a rank 2 layout:
```
layout(i, j) = i * stride_0 + j * stride_1
```

No runtime logic. No Branches. Just Math.

### Creating layouts in CuTe

A. Shape
```
using namespace cute;

auto shape = make_shape(4, 8);
```

This means:

Logical tensor of shape (4 rows, 8 columns)

Important:
- Rank = 2
- Fully compile-time
- Just describes what exists, not memory

B. Stride

Row Major matrix:
```
row stride = number of columns
col stride = 1
```
```
auto stride = make_stride(8, 1);
```

C. Layout = Shape + Stride
```
auto layout = make_layout(shape, stride);
```

This layout represents a 4x8 row-major matrix

### Layout as a function

A layout is callable
```
int offset = layout(2, 3);
```

This expands to:
```
offset = 2 * 8 + 3 * 1 = 19
```
That's the only thing a layout does.
> CuTe is powerful because layouts are values, not conventions.

### Verifying at Compile Time

You can query layout properties:
```
static_assert(rank(layout) == 2);
static_assert(size<0>(layout) == 4);
static_assert(size<1>(layout) == 8);
```

This matters because:
- CuTe algorithms branch on rank
- Tile shapes are validated at compile time
- Bugs surface early

### Column Major Example (Same Shape, Different Layout)

Column Major Matrix:
```
auto col_major = make_layout(
    make_shape(4, 8),
    make_stride(1, 4)
);
```
Now:
```
col_major(i, j) = i * 1 + j + 4
```

Same logical tensor. Different memory walk.

Your indexing code doesn't change.

### Why this is huge for GPU kernels
In CUDA without CuTe, you'd write:
```
ptr[i * ld + j]
```

Now imagine:
- shared memory tiles
- transposed tiles
- swizzled fragments
- MMA layouts

Without CuTe -> nightmare

With CuTe -> swap layouts, keep code

### Minimal Kernel exmaple

```
__global__ void add_one(float* A) {
    using namespace cute;
    
    auto layout = make_layout(
        make_shape(4, 8),
        make_stride(8, 1)
    );

    auto tensor = make_tensor(A, layout);

    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i < 4 && j < 8) {
        tensor(i, j) += 1.0f;
    }
}
```

Important:
- `tensor(i, j)` -> pointer arithmetic
- Comiler fully inlines it
- No overhead vs hand-written indexing

### Mental Model to lock-in
> Layout is a pure, compile-time function that maps logical indices to memory offsets.

Everything else in CuTe:
- layout algebra
- tensor partitioning
- MMA tiling

these all are just layout transformation.

### Exercise 1

1. Create a column-major layout for shape `(3, 5)`
2. Compute offsets for:
    - `(0, 0)`
    - `(1, 0)`
    - `(0, 1)`
    - `(2, 4)`
3. WRite the expected numeric offsets in comments
4. Verify using `layout(i, j)`

Skeleton:
```
using namespace cute;

auto layout = make_layout(
    make_shape(3, 5),
    make_stride(?, ?)
);

// Expected:
// layout(0,0) = ?
// layout(1,0) = ?
// layout(0,1) = ?
// layout(2,4) = ?
```

Solution:
```
using namespace cute;

// for layout (0, 0)
auto layout = make_layout(
    make_shape(3, 5),
    make_stride(1, 3)
);

// verify using layout(i, j)
static_assert(layout(0, 0) == 0);
static_assert(layout(1, 0) == 1);
static_assert(layout(0, 1) == 3);
static_assert(layout(2, 4) == 14);

```
### Exercise 2

> If two layouts have the same shape but different strides, do they represent the same tensor or different tensors?

Ans: Same shape + different strides = same logical tensor, different layouts (views)
