# Layout Algebra

### What is Layout Algebra?

> Layout Algebra = manipulating index -> offset functions using math, not loops.

CuTe treats layouts as pure functions.

Algebra means:
- you can compose them
- you can nest them
- you can factor them
- you can reshape without touching memory

This is why CuTe scales from simple matrices -> MMA -> FlashAttention

### Layouts Are Functions (Formal View)

A layout is a function:
```
L : (i_0, i_1, ..., i_n) -> linear offset
```
Example:
```
auto L = make_layout(make_shape(4, 8), make_stride(8, 1));
```
Mathematically:
```
L(i, j) = 8 * i + j
```
This matters bercause functions compose.

### Composition; `compose(A, B)`

In Math:
```
(A ∘ B)(i, j) = A(B(x))
```
In CuTe:
> Apply layout B first, then layout A

This is the single most important operation in CuTe.

Why composition exists

We often want to:
- tile a tensor
- then map tiles into memory
- them map threads into tiles

Instead of writing index math:
```
(((i / tile) * something) + (i % tile))
```
CuTe does:
```
compose(memory_layout, tile_layout)
```

Simple Example (1D -> 1D)

Step 1: A base layout
```
auto base = make_layout(
    make_shape(16),
    make_stride(1)
);
```
This maps:
```
i -> 1
```

Step 2: A tiling layout

Let's tile 16 elements into `(4 tiles x 4 elements)`
```
auto tile = make_layout(
    make_shape(4, 4),
    make_stride(4, 1)
);
```

This maps:
```
(tile_id, intra_tile) -> tile_id*4 + intra_tile
```

Step 3: Compose

```
auto composed = compose(base, tile);
```
Now:
```
composed(t, k) = base(tile(t, t)) = t*4 + k
```
- Same memory
- Different index space
- Zero runtime cost

> Composition changes how you index, not where data lives.

### `product()` - Increasing Rank

What does `product()` do?

`product(a, b)` creates a higher-rank layout by combining two layouts

Think:
```
product -> Cartesian product of index spaces
```

Example: Build 2D layout from 1D pieces

Row Layout
```
auto row = make_layout(
    make_shape(4),
    make_stride(8)
);
```
Column Layout
```
auto col = make_layout(
    make_shape(8),
    make_stride(1)
);
```

Product
```
auto mat = product(row, col);
```
This creates a rank-2 layout:
```
mat(i, j) = row(i) + col(j)
          = i * 8 + j
```

Which is exactly a 4x8 row-major matrix

Why `product` exists?

Because CuTe builds layouts dimension-by-dimension

This enables:
- hierarchical tiling
- warp-level layouts
- MMA Fragment layouts

without rewriting math

### Tiling = Product + Composition

This is where CuTe becomes beautiful. Let's tile a matrix:

Logical indent
```
Matrix -> tiles -> elements
```

CuTe Expression:
```
compose(global_layout, tile_layout)
```

Example: 8x8 matrix Tiled in 4x4 Blocks
```
auto global = make_layout(
    make_shape(8, 8),
    make_stride(8, 1)
);
```

Tile Layout (2x2 tiles of 4x4)
```
auto tiles = make_layout(
    make_shape(2, 2, 4, 4),
    make_stride(32, 4, 8, 1)
);
```

Interpretation:
- `(tile_i, tile_j)` select tile
- `(i, j)` select element inside tile

Compose
```
auto tiled = compose(global, tiles);
```
Now:
```
tiled(ti, tj, i, j)
```

Indexes into the same matrix, but through a tiled view

No index arithmetic, no branches, compiler sees everything.

### Exercise 1

```
auto A = make_layout(
    make_shape(8),
    make_stride(1)
);

auto B = make_layout(
    make_shape(2, 4),
    make_stride(4, 1),
);
```

Answer:
1. What does `B(i, j)` compute?
> `B(i, j) = i * 4 + j` 

2. What does `compose(A, B)(i, j)` compute?
> `A(B(i, j)) = A(i * 4 + j) = (i * 4 + j) * 1 = i * 4 + j`

3. What is the maximum offset produced?
> 7

### Exercise 2
```
auto C = compose(A, B);

static_assert(C(1, 3) == ?);
static_assert(C(0, 2) == ?);
static_assert(C(1, 0) == ?);
```

Answer:

```
auto C = compose(A, B);

static_assert(C(1, 3) == 7);
static_assert(C(0, 2) == 2);
static_assert(C(1, 0) == 4);
```

### Exercise 3

> Layout composition is useful because it allows us to __ without __

Answer:
Layout composition is useful because it allows us to *change index spaces* without *changing memory layout*.

