// Layer 1 driver. FP16 inputs, FP32 accumulate+output. Verifies each kernel
// against cublas_gemm_fp16_fp32 (COMPUTE_32F), benchmarks with median+stddev.

#include "common/bench_harness.h"
#include "mp_gemm_common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Forward decls from each variant TU.
void wmma_naive_launch(const MpGemmParams& p);

namespace {

struct Args {
    int M = 2048;
    int N = 2048;
    int K = 2048;
    int warmup = 3;
    int iters = 50;
    int runs = 5;
    unsigned seed = 0xC0FFEEu;
    std::string kernel = "all";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if      (k == "--M")      a.M      = std::atoi(next("--M"));
        else if (k == "--N")      a.N      = std::atoi(next("--N"));
        else if (k == "--K")      a.K      = std::atoi(next("--K"));
        else if (k == "--warmup") a.warmup = std::atoi(next("--warmup"));
        else if (k == "--iters")  a.iters  = std::atoi(next("--iters"));
        else if (k == "--runs")   a.runs   = std::atoi(next("--runs"));
        else if (k == "--seed")   a.seed   = (unsigned)std::strtoul(next("--seed"), nullptr, 0);
        else if (k == "--kernel") a.kernel = next("--kernel");
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); std::exit(2); }
    }
    return a;
}

double gflops_of(int M, int N, int K, double median_ms) {
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    return (flops / 1.0e9) / (median_ms / 1.0e3);
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const int M = args.M, N = args.N, K = args.K;

    if (M % WMMA_M || N % WMMA_N || K % WMMA_K) {
        std::fprintf(stderr, "M, N, K must be multiples of %d for WMMA\n", WMMA_M);
        return 2;
    }

    const size_t sA = (size_t)M * K, sB = (size_t)K * N, sC = (size_t)M * N;

    std::vector<half>  hA(sA), hB(sB);
    std::vector<float> hRef(sC), hGot(sC);
    fill_uniform_half(hA.data(), hA.size(), -1.0f, 1.0f, args.seed ^ 0xA1u);
    fill_uniform_half(hB.data(), hB.size(), -1.0f, 1.0f, args.seed ^ 0xB2u);

    half  *dA, *dB;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, sA * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, sB * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, sC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sA * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sB * sizeof(half),  cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    MpGemmParams p{M, N, K, dA, dB, dC};

    // Reference on this exact shape.
    cublas_gemm_fp16_fp32(handle, p);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, sC * sizeof(float), cudaMemcpyDeviceToHost));

    // Registry (insertion order = story order).
    KernelRegistry<MpGemmLaunch> registry;
    registry.emplace_back("wmma_naive", wmma_naive_launch);

    // Audit §L1.1.1: tol=1.0 is too loose; atol+rtol at FP16-scale margins.
    const float atol = 0.5f;
    const float rtol = 1e-3f;

    std::printf("%-16s  %10s  %10s  %7s  %8s  %s\n",
                "kernel", "median(ms)", "stddev(ms)", "min(ms)", "GFLOPS", "verify");
    std::printf("%-16s  %10s  %10s  %7s  %8s  %s\n",
                "----------------", "----------", "----------", "-------", "------", "------");

    for (auto& entry : registry) {
        const std::string& name = entry.first;
        MpGemmLaunch launch = entry.second;
        if (args.kernel != "all" && args.kernel != name) continue;

        poison_output(dC, sC);
        launch(p);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hGot.data(), dC, sC * sizeof(float), cudaMemcpyDeviceToHost));
        const bool pass = verify_close<float>(hRef.data(), hGot.data(), sC, atol, rtol);

        BenchStats stats = benchmark_kernel(
            [&]() {
                poison_output(dC, sC);
                launch(p);
            },
            args.warmup, args.iters, args.runs);

        std::printf("%-16s  %10.4f  %10.4f  %7.4f  %8.2f  %s\n",
                    name.c_str(),
                    stats.median_ms, stats.stddev_ms, stats.min_ms,
                    gflops_of(M, N, K, stats.median_ms),
                    pass ? "PASS" : "FAIL");
    }

    if (args.kernel == "all" || args.kernel == "cublas") {
        BenchStats stats = benchmark_kernel(
            [&]() {
                poison_output(dC, sC);
                cublas_gemm_fp16_fp32(handle, p);
            },
            args.warmup, args.iters, args.runs);

        std::printf("%-16s  %10.4f  %10.4f  %7.4f  %8.2f  %s\n",
                    "cublas",
                    stats.median_ms, stats.stddev_ms, stats.min_ms,
                    gflops_of(M, N, K, stats.median_ms),
                    "ref");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
