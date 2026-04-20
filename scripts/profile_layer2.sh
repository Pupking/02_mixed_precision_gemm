#!/usr/bin/env bash
# Layer-1 (mixed-precision GEMM) profile capture. Thin wrapper around
# scripts/profile_layer.sh. See that file for the parameter contract.

set -euo pipefail
cd "$(dirname "$0")/.."

LAYER=02_mixed_precision_gemm \
BENCH=./build/bin/mp_gemm_bench \
SHAPE="--M 2048 --N 2048 --K 2048" \
KERNELS="wmma_naive wmma_shared wmma_cp_async cublas" \
    ./scripts/profile_layer.sh
