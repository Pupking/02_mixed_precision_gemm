// Layer 1 — WMMA + cp.async double-buffer (sm_80+).
// Two SMEM buffers: while WMMA computes on buf[cur], cp.async.cg.16
// streams the next K-tile into buf[next]. BK halved 64->32 vs shared
// to fit two buffers inside ~30 KB so 3 blocks still resident per SM.
// ptxas sm_86 <4,2,2,2,32>: regs=79 smem=29696 spill=0, 3 blocks/SM
//   (79 * 256 = 20224 regs/block; 65536 / 20224 = 3; SMEM 100 KB / 29696 = 3)
// ncu anchor: smsp__warp_issue_stalled_long_sb_per_issue_active.pct (drop vs shared)
//           + smsp__inst_executed_pipe_cpasync.sum (>0 = cp.async path active)

#include "mp_gemm_common.h"
#include "common/cuda_primitives.cuh"

#include <mma.h>
#include <cassert>
#include <cstdint>

using namespace nvcuda;

namespace {

constexpr int WARP_SIZE = 32;

template <int WARPS_M, int WARPS_N, int WTILE_M, int WTILE_N, int BK_>
__global__ __launch_bounds__(WARPS_M * WARPS_N * WARP_SIZE)
void wmma_cp_async_kernel(const half* __restrict__ A,
                          const half* __restrict__ B,
                          float*      __restrict__ C,
                          int M, int N, int K) {
    constexpr int NUM_WARPS = WARPS_M * WARPS_N;
    constexpr int NTHREADS  = NUM_WARPS * WARP_SIZE;
    constexpr int BM = WARPS_M * WTILE_M * WMMA_M;
    constexpr int BN = WARPS_N * WTILE_N * WMMA_N;
    constexpr int BK = BK_;
    constexpr int K_STEPS = BK / WMMA_K;

    static_assert(BK % WMMA_K == 0, "BK must be a multiple of WMMA_K");
    constexpr int SA_STRIDE = BK + 8;
    constexpr int SB_STRIDE = BN + 8;
    static_assert((SA_STRIDE & 7) == 0, "WMMA half LD must be multiple of 8");
    static_assert((SB_STRIDE & 7) == 0, "WMMA half LD must be multiple of 8");
    // Audit §L1.1.2: cp.async.cg.16 needs SMEM row stride aligned to 16 bytes.
    static_assert((SA_STRIDE * sizeof(half)) % 16 == 0, "SA row stride must be 16B-aligned");
    static_assert((SB_STRIDE * sizeof(half)) % 16 == 0, "SB row stride must be 16B-aligned");

    __shared__ half sA[2][BM * SA_STRIDE];
    __shared__ half sB[2][BK * SB_STRIDE];

    const int tid     = threadIdx.x;
    const int warpId  = tid / WARP_SIZE;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;

    const int brow = blockIdx.y * BM;
    const int bcol = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WTILE_M][WTILE_N];
    #pragma unroll
    for (int i = 0; i < WTILE_M; i++)
        #pragma unroll
        for (int j = 0; j < WTILE_N; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    constexpr int A_CHUNKS = (BM * BK) / 8;
    constexpr int B_CHUNKS = (BK * BN) / 8;

    auto load_tile_async = [&](int buf, int k) {
        #pragma unroll
        for (int idx = tid; idx < A_CHUNKS; idx += NTHREADS) {
            int r  = idx / (BK / 8);
            int c  = (idx % (BK / 8)) * 8;
            int gr = brow + r;
            int gc = k + c;

            half* dst = &sA[buf][r * SA_STRIDE + c];
            if (gr < M && gc + 7 < K) {
                cp_async_16B(dst, &A[gr * K + gc]);
            } else {
                reinterpret_cast<int4*>(dst)[0] = make_int4(0, 0, 0, 0);
            }
        }

        #pragma unroll
        for (int idx = tid; idx < B_CHUNKS; idx += NTHREADS) {
            int r  = idx / (BN / 8);
            int c  = (idx % (BN / 8)) * 8;
            int gr = k + r;
            int gc = bcol + c;

            half* dst = &sB[buf][r * SB_STRIDE + c];
            if (gr < K && gc + 7 < N) {
                cp_async_16B(dst, &B[gr * N + gc]);
            } else {
                reinterpret_cast<int4*>(dst)[0] = make_int4(0, 0, 0, 0);
            }
        }

        cp_async_commit();
    };

    auto compute_tile = [&](int buf) {
        #pragma unroll
        for (int ks = 0; ks < K_STEPS; ks++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WTILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WTILE_N];

            #pragma unroll
            for (int i = 0; i < WTILE_M; i++) {
                int aRow = (warpRow * WTILE_M + i) * WMMA_M;
                wmma::load_matrix_sync(a_frag[i],
                    &sA[buf][aRow * SA_STRIDE + ks * WMMA_K], SA_STRIDE);
            }
            #pragma unroll
            for (int j = 0; j < WTILE_N; j++) {
                int bCol = (warpCol * WTILE_N + j) * WMMA_N;
                wmma::load_matrix_sync(b_frag[j],
                    &sB[buf][ks * WMMA_K * SB_STRIDE + bCol], SB_STRIDE);
            }

            #pragma unroll
            for (int i = 0; i < WTILE_M; i++)
                #pragma unroll
                for (int j = 0; j < WTILE_N; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
    };

    const int num_k_tiles = (K + BK - 1) / BK;

    load_tile_async(0, 0);

    for (int t = 0; t < num_k_tiles - 1; t++) {
        int cur  = t & 1;
        int next = 1 - cur;

        load_tile_async(next, (t + 1) * BK);

        cp_async_wait_group<1>();
        __syncthreads();

        compute_tile(cur);

        __syncthreads();
    }

    {
        int cur = (num_k_tiles - 1) & 1;
        cp_async_wait_group<0>();
        __syncthreads();
        compute_tile(cur);
    }

    #pragma unroll
    for (int i = 0; i < WTILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < WTILE_N; j++) {
            int outRow = brow + (warpRow * WTILE_M + i) * WMMA_M;
            int outCol = bcol + (warpCol * WTILE_N + j) * WMMA_N;
            if (outRow < M && outCol < N)
                wmma::store_matrix_sync(C + outRow * N + outCol,
                    c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

} // namespace

void wmma_cp_async_launch(const MpGemmParams& p) {
    constexpr int WARPS_M = 4, WARPS_N = 2;
    constexpr int WTILE_M = 2, WTILE_N = 2;
    constexpr int BK      = 32;
    constexpr int BM = WARPS_M * WTILE_M * WMMA_M;
    constexpr int BN = WARPS_N * WTILE_N * WMMA_N;
    constexpr int NTHREADS = WARPS_M * WARPS_N * WARP_SIZE;

    // Audit §L1.1.2: cp.async.cg.16 source-pointer alignment.
    assert((reinterpret_cast<uintptr_t>(p.dA) % 16) == 0);
    assert((reinterpret_cast<uintptr_t>(p.dB) % 16) == 0);
    assert((p.K % 8) == 0);

    auto kernel = wmma_cp_async_kernel<WARPS_M, WARPS_N, WTILE_M, WTILE_N, BK>;

    static bool opt_in_done = false;
    if (!opt_in_done) {
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        opt_in_done = true;
    }

    dim3 block(NTHREADS);
    dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);
    kernel<<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    CUDA_CHECK_LAST();
}
