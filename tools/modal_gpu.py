"""Runs Katal's GPU tests on a Modal T4.

    modal run tools/modal_gpu.py

Needs MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in the environment, or a token
from `modal token new`. The checkout is uploaded as it stands, built in the
container against the CUDA and cuTENSOR it installs, and `ctest -L gpu` runs
there. A failing test makes `modal run` exit nonzero.
"""

import pathlib
import subprocess

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent

# The CI compile check uses CUDA 12.6 too. The devel image carries nvcc, which
# FindCUDAToolkit uses to locate the toolkit, and the cudart headers. git and
# ca-certificates let CMake fetch enumstr. cuTENSOR comes from the same script
# and pin as the CI compile check; FindcuTENSOR reads CUTENSOR_ROOT.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install(
        "cmake", "ninja-build", "g++", "catch2", "git", "ca-certificates",
    )
    .add_local_file(
        REPO / "tools" / "install-cutensor.sh",
        "/opt/install-cutensor.sh",
        copy=True,
    )
    .run_commands("bash /opt/install-cutensor.sh /opt")
    .env({"CUTENSOR_ROOT": "/opt/cutensor"})
    .add_local_dir(
        REPO,
        "/src",
        ignore=[".git", "build", "build-*", "docs-build", "docs/api"],
    )
)

app = modal.App("katal-gpu-tests", image=image)


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


@app.function(gpu="T4", timeout=30 * 60)
def gpu_tests() -> None:
    run("nvidia-smi")
    run(
        "cmake", "-S", "/src", "-B", "/tmp/build", "-GNinja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DKATAL_WARNINGS_AS_ERRORS=ON",
    )
    run("cmake", "--build", "/tmp/build")
    # Without cuTENSOR the gpu tests are never built, and ctest would pass
    # having run nothing; --no-tests=error makes that a failure.
    run(
        "ctest", "--test-dir", "/tmp/build", "--output-on-failure",
        "--label-regex", "gpu", "--no-tests=error",
    )


@app.local_entrypoint()
def main() -> None:
    gpu_tests.remote()
