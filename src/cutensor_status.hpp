#ifndef KATAL_SRC_CUTENSOR_STATUS_HPP_
#define KATAL_SRC_CUTENSOR_STATUS_HPP_

#include <cuda_runtime.h>
#include <cutensor.h>

#include <katal/error.hpp>

namespace katal::util {

/// The Errc a failing cuTENSOR status is reported as.
inline Errc errc_of(cutensorStatus_t status) noexcept {
  switch (status) {
    case CUTENSOR_STATUS_NOT_SUPPORTED:
      return Errc::UnsupportedPattern;
    case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE:
      return Errc::WorkspaceTooSmall;
    default:
      return Errc::CutensorError;
  }
}

}  // namespace katal::util

/// Turns a cuTENSOR status into a katal::Error. The equation argument is
/// required rather than optional because it is what makes a cuTENSOR failure
/// diagnosable.
#define KATAL_CHECK(expr, equation)                                           \
  do {                                                                        \
    const cutensorStatus_t katal_status_ = (expr);                            \
    if (katal_status_ != CUTENSOR_STATUS_SUCCESS)                             \
      throw ::katal::Error(::katal::util::errc_of(katal_status_), (equation), \
                           cutensorGetErrorString(katal_status_));            \
  } while (false)

/// The same for a CUDA runtime call.
#define KATAL_CHECK_CUDA(expr, equation)                         \
  do {                                                           \
    const cudaError_t katal_cuda_ = (expr);                      \
    if (katal_cuda_ != cudaSuccess)                              \
      throw ::katal::Error(::katal::Errc::CudaError, (equation), \
                           cudaGetErrorString(katal_cuda_));     \
  } while (false)

#endif  // KATAL_SRC_CUTENSOR_STATUS_HPP_
