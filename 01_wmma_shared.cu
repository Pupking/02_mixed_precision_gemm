// Layer 1 — WMMA + SMEM staging + 2x2 WMMA-tile register block per warp.
// Block = 8 warps (WARPS_M=4, WARPS_N=2). Per-block output = 128x64.
// ptxas sm_86 <4,2,2,2,64>: regs=77 smem=27648 spill=0, 3 blocks/SM
//   (77 * 256 = 19712 regs/block; 65536 / 19712 = 3; SMEM 100 KB / 27648 = 3)
// ncu anchor: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum (drop vs naive)
//           + l1tex_hit_rate_pct (tile reuse via SMEM raises this)

#include "mp_gemm_common.h"

#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int WARP_SIZE = 32;

template <int WARPS_M, int WARPS_N, int WTILE_M, int WTILE_N, int BK_>
__global__ __launch_bounds__(WARPS_M * WARPS_N * WARP_SIZE)
void wmma_shared_kernel(const half* __restrict__ A,
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

    __shared__ half sA[BM * SA_STRIDE];
    __shared__ half sB[BK * SB_STRIDE];

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

    constexpr int A_PACKED = (BM * BK) / 2;
    constexpr int B_PACKED = (BK * BN) / 2;

    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int idx = tid; idx < A_PACKED; idx += NTHREADS) {
            int r  = idx / (BK / 2);
            int c  = (idx % (BK / 2)) * 2;
            int gr = brow + r;
            int gc = k + c;
            int packed = (gr < M && gc < K)
                ? *reinterpret_cast<const int*>(&A[gr * K + gc])
                : 0;
            *reinterpret_cast<int*>(&sA[r * SA_STRIDE + c]) = packed;
        }

        #pragma unroll
        for (int idx = tid; idx < B_PACKED; idx += NTHREADS) {
            int r  = idx / (BN / 2);
            int c  = (idx % (BN / 2)) * 2;
            int gr = k + r;
            int gc = bcol + c;
            int packed = (gr < K && gc < N)
                ? *reinterpret_cast<const int*>(&B[gr * N + gc])
                : 0;
            *reinterpret_cast<int*>(&sB[r * SB_STRIDE + c]) = packed;
        }

        __syncthreads();

        for (int ks = 0; ks < K_STEPS; ks++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WTILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WTILE_N];

            #pragma unroll
            for (int i = 0; i < WTILE_M; i++) {
                int aRow = (warpRow * WTILE_M + i) * WMMA_M;
                wmma::load_matrix_sync(a_frag[i],
                    &sA[aRow * SA_STRIDE + ks * WMMA_K], SA_STRIDE);
            }
            #pragma unroll
            for (int j = 0; j < WTILE_N; j++) {
                int bCol = (warpCol * WTILE_N + j) * WMMA_N;
                wmma::load_matrix_sync(b_frag[j],
                    &sB[ks * WMMA_K * SB_STRIDE + bCol], SB_STRIDE);
            }

            #pragma unroll
            for (int i = 0; i < WTILE_M; i++)
                #pragma unroll
                for (int j = 0; j < WTILE_N; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }

        __syncthreads();
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

void wmma_shared_launch(const MpGemmParams& p) {
    constexpr int WARPS_M = 4, WARPS_N = 2;
    constexpr int WTILE_M = 2, WTILE_N = 2;
    constexpr int BK      = 64;
    constexpr int BM = WARPS_M * WTILE_M * WMMA_M;
    constexpr int BN = WARPS_N * WTILE_N * WMMA_N;
    constexpr int NTHREADS = WARPS_M * WARPS_N * WARP_SIZE;

    auto kernel = wmma_shared_kernel<WARPS_M, WARPS_N, WTILE_M, WTILE_N, BK>;

    // Audit §L1.3.1: opt-in carveout is a one-time cost, not per-launch.
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
