// Layer 1 — WMMA naive: one warp per 16x16 output tile, no SMEM reuse.
// ptxas sm_86: regs=40 smem=0 spill=0 (128 threads/block, no __shared__)
// ncu anchor: smsp__inst_executed_pipe_tensor_op_hmma.sum (confirm HMMA path)
//           + dram__throughput.avg.pct_of_peak_sustained_elapsed (baseline)

#include "mp_gemm_common.h"

#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int WARPS_PER_BLOCK = 4;   // 128 threads / block

__global__ __launch_bounds__(WARPS_PER_BLOCK * 32)
void wmma_naive_kernel(const half* __restrict__ A,
                       const half* __restrict__ B,
                       float*      __restrict__ C,
                       int M, int N, int K) {
    const int warpId   = threadIdx.y;
    const int tileRow  = blockIdx.y * WARPS_PER_BLOCK + warpId;
    const int tileCol  = blockIdx.x;

    const int rowStart = tileRow * WMMA_M;
    const int colStart = tileCol * WMMA_N;

    if (rowStart >= M || colStart >= N) return;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                  c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + rowStart * K + k,        K);
        wmma::load_matrix_sync(b_frag, B + k        * N + colStart, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + rowStart * N + colStart, c_frag, N, wmma::mem_row_major);
}

} // namespace

void wmma_naive_launch(const MpGemmParams& p) {
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid(p.N / WMMA_N,
              (p.M + WMMA_M * WARPS_PER_BLOCK - 1) / (WMMA_M * WARPS_PER_BLOCK));
    wmma_naive_kernel<<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    CUDA_CHECK_LAST();
}
