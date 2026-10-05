#ifndef KATAL_SRC_CUDA_STATUS_HPP_
#define KATAL_SRC_CUDA_STATUS_HPP_

#include <exception>

#include <cuda_runtime.h>

namespace katal::util {

/// True when `status` means the CUDA context is already unusable: a sticky
/// device fault, which every later call in the process reports again, or the
/// runtime shutting down at exit. Memory cannot be freed then, and need not
/// be, since it goes with the context.
inline bool context_lost(cudaError_t status) noexcept {
  switch (status) {
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorLaunchTimeout:
    case cudaErrorHardwareStackError:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorAssert:
    case cudaErrorECCUncorrectable:
    case cudaErrorExternalDevice:
#if CUDART_VERSION >= 12080
    case cudaErrorTensorMemoryLeak:
#endif
    case cudaErrorContextIsDestroyed:
    case cudaErrorCudartUnloading:
      return true;
    default:
      return false;
  }
}

/// For a release in cleanup, which cannot throw. A failure while the context
/// is healthy would leak or free memory still in use, so it terminates. One
/// after the context is lost only repeats the original fault, which the
/// program is already unwinding from, so it is ignored.
inline void check_release(cudaError_t status) noexcept {
  if (status != cudaSuccess && !context_lost(status)) std::terminate();
}

}  // namespace katal::util

#endif  // KATAL_SRC_CUDA_STATUS_HPP_
