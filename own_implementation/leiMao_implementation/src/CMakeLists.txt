cmake_minimum_required(VERSION 3.22.1)

project(CUDA-GEMM-Algorithms VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Auto-detect GPU architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Try to detect GPU architecture using nvidia-smi
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE NVIDIA_SMI_RESULT
    )
    
    if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
        # Convert "8.9" to "89"
        string(REPLACE "." "" GPU_ARCH "${GPU_COMPUTE_CAP}")
        # Handle multiple GPUs - take the first one
        string(REGEX REPLACE "\n.*" "" GPU_ARCH "${GPU_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
        message(STATUS "Auto-detected GPU architecture: ${GPU_ARCH}")
    else()
        # Fallback to common architectures
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
        message(STATUS "Could not detect GPU, using fallback architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_path(CUDA_GEMM_INCLUDE_DIRS cuda_gemm_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
file(GLOB CUDA_GEMM_HEADERS ${CMAKE_SOURCE_DIR}/include/*.hpp ${CMAKE_SOURCE_DIR}/include/*.cuh)

# Add all the source files in the current directory to build the library
add_library(
    cuda_gemm 
    SHARED 
    cuda_gemm_utils.cu 
    # 00_non_coalesced_global_memory_access.cu
    # 01_coalesced_global_memory_access.cu
    02_2d_block_tiling.cu
    # 02_2d_block_tiling_vectorized_memory_access.cu
    # 03_2d_block_tiling_1d_thread_tiling.cu
    # 03_2d_block_tiling_1d_thread_tiling_vectorized_memory_access.cu
    # 04_2d_block_tiling_2d_thread_tiling.cu
    # 04_2d_block_tiling_2d_thread_tiling_vectorized_memory_access.cu
    # 05_2d_block_tiling_2d_thread_tiling_matrix_transpose.cu
    # 05_2d_block_tiling_2d_thread_tiling_matrix_transpose_vectorized_memory_access.cu
    # 06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose.cu
    # 06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_vectorized_memory_access.cu
    # 06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_vectorized_memory_access_double_buffered.cu
    # 07_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_wmma.cu
    # 07_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_wmma_vectorized_memory_access.cu
    # 07_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_wmma_vectorized_memory_access_double_buffered.cu
)

# Add the include directory of the library to the include directories of the project
target_include_directories(cuda_gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})

# No need to set CUDA_ARCHITECTURES per-target, CMAKE_CUDA_ARCHITECTURES applies globally
install(TARGETS cuda_gemm DESTINATION lib)
install(FILES ${CUDA_GEMM_HEADERS} DESTINATION include)

add_executable(profile_cuda_gemm_fp32 profile_cuda_gemm_fp32.cu)
target_link_libraries(profile_cuda_gemm_fp32 cuda_gemm CUDA::cublas)

add_executable(profile_cuda_gemm_fp16 profile_cuda_gemm_fp16.cu)
target_link_libraries(profile_cuda_gemm_fp16 cuda_gemm CUDA::cublas)