#!/usr/bin/env bash
# Installs the pinned cuTENSOR into <prefix> for CI and the Modal GPU image,
# and prints the root to pass as CUTENSOR_ROOT or -DKATAL_CUTENSOR_ROOT.
#
#     tools/install-cutensor.sh <prefix>
#
# The apt package libcutensor-dev in NVIDIA's repository stopped at 2.2.0, so
# this unpacks NVIDIA's PyPI wheel instead, which pins an exact release. The
# wheel ships only the versioned libcutensor.so.2, so the unversioned link
# that find_library looks for is added here.
set -euo pipefail

readonly version=2.8.1
readonly cuda_major=12
readonly prefix="${1:?usage: install-cutensor.sh <prefix>}"

work="$(mktemp -d)"
readonly work
trap 'rm -rf "${work}"' EXIT

python3 -m pip download --quiet --no-deps --only-binary=:all: \
  --dest "${work}" "cutensor-cu${cuda_major}==${version}"
mkdir -p "${prefix}"
python3 -m zipfile -e "${work}"/cutensor_cu${cuda_major}-*.whl "${prefix}"

readonly root="${prefix}/cutensor"
ln -sf libcutensor.so.2 "${root}/lib/libcutensor.so"
echo "${root}"
