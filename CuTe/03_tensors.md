# Tensors

### What is a Tensor in CuTe?

```
Tensor = (pointer, layout)
```
That's it.

CuTe tensors are views, not containers.

Compare with PyTorch
| PyTorch | CuTe |
|--- |--- |
| Own Memory | Never owns memory |
| Dynamic Shape | Compile-time shape | 
| Runtime Strides | Compile-time strides |
| Heavt abstraction | Zero-cost abstraction |

CuTe tensors exist to make kernels readable and correct, not to manage memory.

### Creating a Tensor

Let's bind a pointer to a layout:
```
float* ptr = ...; // global or shared memory

auto layout = make_layout(
    make_shape(4, 8),
    make_stride(8, 1)
);

auto tensor = make_tensor(ptr, layout);
```
Now:
```
tensor(i, j) = *(ptr + layout(i, j))
```
The compiler literally emits pointer arithmetic

### Tensor Indexing
```
float x = tensor(2, 3);
```
Expands to:
```
*(ptr + 2 * 8 + 3)
```
GPU Insight

Because:
- layout is constexpr
- indexing is inlined

### Tensor Rank comes from layout

Tensor rank is inherited directly from the layout
```
rank(tensor) == rank(layout)
```
Examples:
- layout(8) -> vector tensor
- layout(4, 8) -> matrix tensor
- layout(2, 2, 4, 4) -> tiled tensor

CuTe kernels are rank-generic

### Slicing `tensor(_, j)` (Views, not copies)
This is where CuTe starts feeling powerful

Example: Column Slice
```
auto col = tensor(_, 3);
```
What happens?
- `_` means "vary this dimension"
- fixed `j=3`
- returns a rank-1 tensor

Equivalent math
```
col(i) = tensor(i, 3)
       = *(ptr + i * 8 + 3)
```
Just a new layout

Shape & Stride after slicing

Original:
```
shape = (4, 8)
stride = (8, 1)
```

After slicing column `j=3`:
```
shape = (4)
stride = (8)
```
This matters later for shared memory tiles

### Subtensors are layout transformations

Row Slice
```
auto row = tensor(2, _);
```
Results in:
```
shape = (8)
stride = (1)
```
We didn't:
- change memory
- copy data
- reindex manually

We changed the layout

### Tensor + Layout Algebra = Tiling
This is the payoff

Let's reuse layout composition
```
auto tiled_layout = compose(global_layout, tile_layout);
auto tiled_tensor = make_tensor(ptr, tiled_layout);
```
Now we can write:
```
tiled_tensor(tile_i, tile_j, i, j)
```
And the compiler resolves it to:
```
*(ptr + computed_offset)
```
No indexing math written by us

### Shared memory example
```
__shared__ float smem[128];

auto smem_layout = make_layout(
    make_shape(16, 8),
    make_stride(8, 1)
);

auto smem_tensor = make_tensor(smem, smem_layout);
```
Now:
- You can slice
- You can tile
- You can reinterpret layout

All without touching `smem[]`

### Exercise 1 - Offset Reasoning
Given:
```
float* ptr = ...;

auto layout = make_layout(
    make_shape(3, 5),
    make_stride(5, 1)
);

auto tensor = make_tensor(ptr, layout);
```
1. What address does `tensor(2, 4)` access (in terms of `ptr + offset`)?
> `ptr + 2 * 5 + 4 = ptr + 14`

2. What is the shape and stride of `tensor(1, _)`?
> `shape = (5), stride = (1)`

3. What is the shape and stride of `tensor(_, 3)`?
> `shape = (3), stride = (5)`

### Exercise 2
```
auto row = tensor(1, _);
auto col = tensor(_, 3);

static_assert(rank(row) == ?);
static_assert(rank(col) == ?);

// Fill expected offsets:
static_assert(row(0) == ?);  // offset
static_assert(col(2) == ?);  // offset
```

Answer:
```
auto row = tensor(1, _);
auto col = tensor(_, 3);

static_assert(rank(row) == 1);
static_assert(rank(col) == 1);

// Fill expected offsets:
static_assert(row(0) == 5);  // offset
static_assert(col(2) == 13);  // offset
```
### Exercise 3

> A CuTe tensor is best thought of as ___.

Answer:
A CuTe tensor is best thought of as *a view of a memory defined by a layout*.