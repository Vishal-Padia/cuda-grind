### Plan:

So I was facing a problem where I was able to understand the kernel implementation but I wasn't able to implement it. Meaning it might by my C++ skills or it might be that I haven't genuinely understand the kernel implementation and it's optimizations and why those optimizations. So when I told Sriram this, he suggested that I go through the blog again and then understand algorithm and try to implement it on my own, I won't be able to ge the perfect implementation and I'll run into multiple bugs, and debuging those things would help strengthen my C++ skills and also I will understand each and every optimization quite better. 

"I do not understand what I can not create" -- Richard Feynman

Doing the above thing will probably take me around 15days or a month, but I think this is time well spent, understanding each optimizations and why those optimizations. Once I have a working version of the kernel I can benchmark it against LeiMao's implementation and check if there's any difference in my implementation and his. Also claude told me that this is the best advice someone can give, also before benchmarking the kernel try to profile the kernel using `nsys` or `ncu`. This will help me understand what's lacking in the kernel, meaning is it not parallel enough? does it have poor memory access pattern? etc. So this will help me in deriving my intuition regarding what can be optimized more.

So yes, I'll be doing this for the next 15 days or a month. It will be slow but I'll learn a lot.

# 00: Non-Coalesced Memory Access Implementation

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?

# 01: Coalesced Memory Access Implementation

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?

# 02: 2D Block Tiling Implementation

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?

# 03: 2D Block Tiling and 1D Thread Tiling Implementation

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?

# 04: 2D Block Tiling and 2D Thread Tiling Implementation

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?

# 05: 2D Block Tiling, 2D Thread Tiling, and Matrix Transpose Access Implementation

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?

# 06: 2D Block Tiling, 2D Warp Tiling, 2D Thread Tiling, Matrix Transpose with Vectorized Memory Access, and WMMA Implementation

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?

# 07: 2D Block Tiling, 2D Warp Tiling, 2D Thread Tiling, Matrix Transpose with Vectorized Memory Access, and WMMA Implementation

### 01: Profiling the kernel:

### 02: Benchmarking the kernel:

### 03: Comparing the results with LeiMao's implementation:

### 04: What can be optimized more?
