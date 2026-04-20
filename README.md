# 02_mixed_precision_gemm - WMMA-based FP16 GEMM with FP32 accumulate on Ampere

A three-step walk from a naive WMMA kernel to a cp.async double-buffered
pipeline for FP16 inputs with FP32 accumulation. Each step is justified
by a Nsight Compute counter that moved. The same poison + `atol+rtol`
verify harness from Layer 0 is reused; the reference is cuBLAS
`cublasGemmEx` with `CUBLAS_COMPUTE_32F` (dispatches to a CUTLASS
Tensor-Core kernel on sm_86). Reproducible from the `.ncu-rep` files
in this repo.

Per-kernel profile details: **[docs/02_mixed_precision_gemm.md](../../docs/02_mixed_precision_gemm.md)**.

## Deep Dive Contents

- WMMA fragment model and why a naive implementation underutilises the HMMA pipe.
- SMEM staging as the lever that converts K-stride DRAM reads into reused tile loads.
- Double-buffered cp.async pipeline and the long-scoreboard stall it eliminates.
- The remaining gap to cuBLAS/CUTLASS and where it comes from.

## Results

|                  | wmma_naive | wmma_shared | wmma_cp_async | cuBLAS (CUTLASS) |
|------------------|-----------:|------------:|--------------:|-----------------:|
| **ms @ 2048³**   |      5.58  |       1.84  |      **1.63** |             1.33 |
| **GFLOPS**       |     3,080  |      9,332  |     **10,551**|           12,896 |
| **% cuBLAS**     |        24  |         72  |         **82**|              100 |

| step                                                              | what changed                                                  | counter that moved               | step gain |
|-------------------------------------------------------------------|---------------------------------------------------------------|----------------------------------|-----------|
| [00_wmma_naive.cu](00_wmma_naive.cu)                              | 1 warp per 16×16 tile, K-stride WMMA load from GMEM           | — (baseline)                     | —         |
| [01_wmma_shared.cu](01_wmma_shared.cu)                            | 128×64 SMEM tile, 2×2 WMMA-frag reg block per warp, BK=64     | LDG sectors 134M → 12.6M         | 3.02×     |
| [02_wmma_cp_async.cu](02_wmma_cp_async.cu)                        | double-buffered cp.async.cg.16, BK halved to 32, 3 blocks/SM  | long-scoreboard 3.58% → 0.25%    | 1.13×     |

cuBLAS reference goes through `cublasGemmEx(CUBLAS_COMPUTE_32F,
CUBLAS_GEMM_DEFAULT_TENSOR_OP)` — Tensor Cores enabled, FP32
accumulator throughout, for an apples-to-apples comparison on the
HMMA path. Times are medians of 5 × 20 iterations.

## Experimental Setup

<details>
<summary>Click for more details <code>cudaGetDeviceProperties</code> / <code>cudaDeviceGetAttribute</code> </summary>

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (GA107), sm_86, 16 SMs
- Per-SM: 65,536 registers, 1,536 threads, 100 KB shared memory (48 KB static, 99 KB opt-in), 128 KB unified L1/TEX
- Tensor Cores: 3rd-gen, HMMA.16816 throughput 512 FP16×FP16+FP32 MAC / SM / cycle
- Compute peak (HMMA): 512 × 16 SMs × 2 flops × 1.5 GHz = 24.6 TFLOPS FP16→FP32
- On-chip / off-chip: 1.5 MB L2, 3.68 GB VRAM, 128-bit bus, 192 GB/s peak DRAM
- Toolkit / driver: CUDA 13.0.88, driver 580.82.09, compiled `-O3 --gpu-architecture=sm_86`
- cuBLAS: 13.1.0.3 (ships with CUDA 13.0)
- Shape: M = N = K = 2048. FP16 working set ≈ 16.8 MB total A+B, 16 MB C. Arithmetic intensity ≈ 682 flops/byte (FP16 halves byte cost vs FP32); at 192 GB/s the memory ceiling sits at ≈ 131 TFLOPS, well above the HMMA compute peak. Compute-bound throughout.

</details>

## Summary

**Row 0 → Row 1 — break the LDG ceiling.** `wmma_naive` issues one
LDG per WMMA load; each warp traverses K with a stride-K footprint
that the L1 cannot cache at 2048 across 8 concurrent warps/SM
(L1-TEX hit rate 50%). `wmma_shared` pulls both A and B into SMEM
once per BK=64 slice and reruns the WMMA `load_matrix_sync` from
there; the LDG sector count drops 134M → 12.6M (≈10.7×) and long-
scoreboard stalls fall 55% → 3.6%. HMMA pipe utilisation rises
11.7% → 37.0% because the scheduler finally has eligible warps to
issue each cycle (Issue Active % climbs 11.5 → 42.4).

**Row 1 → Row 2 — kill the remaining stall.** `wmma_shared`'s last
3.6% of long-scoreboard stalls are the LDG→LDS bridge: each tile
still travels through the register file on its way to SMEM.
`wmma_cp_async` replaces that with `cp.async.cg.16` which routes
GMEM → SMEM directly and signals completion via a commit/wait
group. Because the async copy runs concurrently with the previous
tile's WMMA compute, the pipeline hides DRAM latency. BK halves
64 → 32 to fit a second SMEM buffer inside the same 3-block/SM
carveout (2 × (128·40 + 32·72) × 2 bytes = 29.7 KB per block;
3 × 29.7 KB ≤ 100 KB). Long-scoreboard stalls collapse 3.6% → 0.25%,
HMMA pipe utilisation rises to 42.8%, and DRAM throughput climbs
36% → 44% — the pipeline saturates memory harder than its single-
buffered precursor.

**Row 2 → cuBLAS.** CUTLASS reaches HMMA pipe 47.2% at 218 registers
per thread, with L1-TEX hit rate 0% (every load is a `cp.async.cg.4`
or `.cg.16` direct to SMEM, bypassing L1) and 6.3M LDG sectors
(≈2× fewer than ours at larger tile 256×128 vs our 128×64). The gap
is cross-block L2 reuse and a larger register-resident tile that
pushes more FMA per load; the step within a block is exhausted.

## Verification

- Cross-checked element-wise against `cublas_gemm_fp16_fp32` (same
  `CUBLAS_COMPUTE_32F` reference) on every launch via
  `verify_close<float>` with `atol = 0.5f, rtol = 1e-3f`
  (audit §L1.1.1 — the reference used `tol = 1.0f` absolute, which
  at K=2048 with `|ref|` ≈ 11 was ≈10% relative and could miss
  systematic off-by-one in a WMMA fragment).
- Output buffer is poisoned with `0xFF` (NaN under FP32) before every
  launch so a half-written kernel cannot pass by reading stale
  reference data.
- WMMA leading-dim alignment is a compile-time `static_assert` on both
  SMEM strides (§L1.1.3). cp.async source-pointer alignment is checked
  at launch via `assert` (§L1.1.2).

## Reproducing

**Build:**
```bash
rm -rf build && mkdir build && cd build
cmake .. && cmake --build . --parallel
cd ..
```

**Run the Layer-1 sweep (harness timings):**
```bash
./build/bin/mp_gemm_bench --M 2048 --N 2048 --K 2048 \
                         --iters 20 --runs 5 --warmup 3
```

**Capture profiles** (Nsight Compute 2025.3+):
```bash
./scripts/profile_layer2.sh
```
Produces one `.ncu-rep` per kernel under `profiles/02_mixed_precision_gemm/`,
plus anchor-metric CSVs under `profiles/02_mixed_precision_gemm/csv/`.

## Scope

- **Per-block, single-shape FP16→FP32.** Every number is at M = N = K = 2048.
  Square-tall or tall-skinny shapes can shift the bottleneck; cuBLASLt or
  CUTLASS is the right tool for shape-dispatched code. The three variants
  here target a single point.
- **WMMA m16n16k16 only.** No MMA PTX (row-level `mma.sync`), no
  asynchronous tensor loads with `ldmatrix`. Both land on Hopper/Blackwell
  as separate primitives; on sm_86 they're accessible but outside scope.
- **No attention-style fusion.** Layer 4 composes this layer's kernels
  with a softmax pass; the bottleneck there is different (online rescaling
  + row reductions, not WMMA throughput).
