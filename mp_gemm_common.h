#pragma once

// Layer 1 — Mixed-Precision GEMM: common types, host helpers, and references.

#include "common/bench_harness.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <random>

// ---------------------------------------------------------------------------
// cuBLAS error check (duplicated from gemm_common.h to keep layers decoupled)
// ---------------------------------------------------------------------------

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (expr);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                 \
                         static_cast<int>(_s), __FILE__, __LINE__);            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// WMMA tile shape (m16n16k16 fp16→fp32)
// ---------------------------------------------------------------------------

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ---------------------------------------------------------------------------
// Problem + launcher contract
// ---------------------------------------------------------------------------
// Row-major: C[M,N] = A[M,K] * B[K,N]. FP16 inputs, FP32 accumulator + output.
// Every kernel launches via a function matching MpGemmLaunch; registered in
// main.cu's KernelRegistry<MpGemmLaunch>.

struct MpGemmParams {
    int M, N, K;
    const half* dA;   // device, row-major [M, K]
    const half* dB;   // device, row-major [K, N]
    float*      dC;   // device, row-major [M, N]  (FP32 output)
};

using MpGemmLaunch = void (*)(const MpGemmParams&);

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

inline void fill_uniform_half(half* buf, std::size_t n,
                              float lo, float hi, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (std::size_t i = 0; i < n; ++i) buf[i] = __float2half(dist(rng));
}

// ---------------------------------------------------------------------------
// cuBLAS reference — FP16 inputs, FP32 accumulator (audit §L1.2.2)
// ---------------------------------------------------------------------------
// cuBLAS is column-major. For row-major C = A*B compute the equivalent
// column-major C^T = B^T * A^T by swapping (A,B) and M↔N. COMPUTE_32F
// pins the accumulator precision; DEFAULT_TENSOR_OP lets cuBLAS pick an
// HMMA algo per shape but the accumulator is FP32 throughout, which makes
// the numerical result shape-deterministic for our tolerance.

inline void cublas_gemm_fp16_fp32(cublasHandle_t handle, const MpGemmParams& p) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        p.N, p.M, p.K,
        &alpha,
        p.dB, CUDA_R_16F, p.N,
        p.dA, CUDA_R_16F, p.K,
        &beta,
        p.dC, CUDA_R_32F, p.N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
