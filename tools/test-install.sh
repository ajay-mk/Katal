#!/usr/bin/env bash
# Installs Katal into a throwaway prefix and builds test/consumer against it.
# That is the only check that the install rules, the exported config and the
# pimpl all hold: the consumer is compiled by a plain host compiler and fails
# to compile if a CUDA header reaches it. CMAKE_CXX_STANDARD in the environment,
# if set, builds both as that standard.
set -euo pipefail

readonly source_dir="$(cd "$(dirname "$0")/.." && pwd)"
readonly work="$(mktemp -d)"
readonly standard=(${CMAKE_CXX_STANDARD:+"-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"})
trap 'rm -rf "${work}"' EXIT

cmake -S "${source_dir}" -B "${work}/build" -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKATAL_BUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="${work}/prefix" "${standard[@]}" >/dev/null
cmake --build "${work}/build" >/dev/null
cmake --install "${work}/build" >/dev/null

cmake -S "${source_dir}/test/consumer" -B "${work}/consumer" -GNinja \
  -DCMAKE_PREFIX_PATH="${work}/prefix" "${standard[@]}" >/dev/null
cmake --build "${work}/consumer" >/dev/null

"${work}/consumer/consumer"
