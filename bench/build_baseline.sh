#!/usr/bin/env bash
# Builds bench/baseline.cu on Tinkercliffs. cuTENSOR has no module of its own
# there; NVHPC ships it, so the paths come from $NVHPC.
set -euo pipefail

module load CUDA/12.6.0
module load NVHPC/25.1-CUDA-12.6.0

readonly math="${NVHPC}/Linux_x86_64/25.1/math_libs/12.6/targets/x86_64-linux"
readonly libdir="$([ -f "${math}/lib/libcutensor.so" ] && echo "${math}/lib" \
                   || echo "${math}/lib64")"

readonly src="$(dirname "$0")/baseline.cu"
readonly out="${1:-./baseline}"

nvcc -O3 -std=c++17 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90 \
  "${src}" -o "${out}" \
  -I"${math}/include" -L"${libdir}" -lcutensor -Xlinker -rpath="${libdir}"

echo "built ${out}"
