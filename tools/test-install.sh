#!/usr/bin/env bash
# Installs DeviceEin into a throwaway prefix and builds test/consumer against it.
# That is the only check that the install rules, the exported config and the
# pimpl all hold: the consumer is compiled by a plain host compiler and fails
# to compile if a CUDA header reaches it.
set -euo pipefail

readonly source_dir="$(cd "$(dirname "$0")/.." && pwd)"
readonly work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

cmake -S "${source_dir}" -B "${work}/build" -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDEVICEEIN_BUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="${work}/prefix" >/dev/null
cmake --build "${work}/build" >/dev/null
cmake --install "${work}/build" >/dev/null

cmake -S "${source_dir}/test/consumer" -B "${work}/consumer" -GNinja \
  -DCMAKE_PREFIX_PATH="${work}/prefix" >/dev/null
cmake --build "${work}/consumer" >/dev/null

"${work}/consumer/consumer"
