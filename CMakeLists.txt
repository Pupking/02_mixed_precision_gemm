cmake_minimum_required(VERSION 3.20)

# RTX 3050 Laptop (GA107) = sm_86. Override via -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(mixed_precision_gemm
    LANGUAGES CXX CUDA
    DESCRIPTION "Mixed-precision (fp16 in, fp32 accum) WMMA GEMM: naive -> SMEM -> cp.async double buffer"
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDAToolkit REQUIRED)

add_library(portfolio_common INTERFACE)
target_include_directories(portfolio_common INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(portfolio_common INTERFACE
    CUDA::cudart
    CUDA::cublas
)

add_executable(mp_gemm_bench
    main.cu
    00_wmma_naive.cu
    01_wmma_shared.cu
    02_wmma_cp_async.cu
)

target_link_libraries(mp_gemm_bench PRIVATE portfolio_common)

# Layer 1 uses cuBLAS (COMPUTE_32F) + WMMA intrinsics (linked via CUDA::cudart).
