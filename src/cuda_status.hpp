#ifndef KATAL_SRC_CUDA_STATUS_HPP_
#define KATAL_SRC_CUDA_STATUS_HPP_

#include <exception>

#include <cuda_runtime.h>

namespace katal::util {

/// True when `status` means the CUDA context is already unusable: a sticky
/// or contained device fault, which every later call in the process reports
/// again, or the runtime shutting down at exit. Memory cannot be freed then,
/// and need not be, since it goes with the context.
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
      // cudaErrorContained (226), a contained device fault, is missing from
      // older headers, so it is matched by value.
      return static_cast<int>(status) == 226;
  }
}

/// For a release in cleanup, which cannot throw. A failure while the context
/// is healthy would leak or free memory still in use, so it terminates. One
/// after the context is lost only repeats the original fault, which the
/// program is already unwinding from, so it is ignored.
inline void check_release(cudaError_t status) noexcept {
  if (status != cudaSuccess && !context_lost(status)) std::terminate();
}

/// Makes `device` current for its lifetime and restores the previous device
/// after, for cleanup that cuTENSOR requires on the device its handle was
/// created on. Failures are ignored: a destructor cannot report them, and a
/// lost context fails the cleanup anyway.
class ScopedDevice {
 public:
  explicit ScopedDevice(int device) noexcept {
    if (cudaGetDevice(&previous_) != cudaSuccess || previous_ == device) {
      previous_ = -1;
      return;
    }
    if (cudaSetDevice(device) != cudaSuccess) previous_ = -1;
  }
  ~ScopedDevice() {
    if (previous_ >= 0) cudaSetDevice(previous_);
  }
  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;

 private:
  int previous_ = -1;
};

}  // namespace katal::util

#endif  // KATAL_SRC_CUDA_STATUS_HPP_
