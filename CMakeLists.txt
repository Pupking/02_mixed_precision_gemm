add_executable(mp_gemm_bench
    main.cu
    00_wmma_naive.cu
)

target_link_libraries(mp_gemm_bench PRIVATE portfolio_common)

# Layer 1 uses cuBLAS (COMPUTE_32F) + WMMA intrinsics (linked via CUDA::cudart).
