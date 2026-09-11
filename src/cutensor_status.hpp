#ifndef KATAL_SRC_CUTENSOR_STATUS_HPP_
#define KATAL_SRC_CUTENSOR_STATUS_HPP_

#include <cuda_runtime.h>
#include <cutensor.h>

#include <katal/error.hpp>

/// Turns a cuTENSOR status into a katal::Error. The equation argument is
/// required rather than optional because it is what makes a cuTENSOR failure
/// diagnosable.
#define KATAL_CHECK(expr, equation)                                  \
  do {                                                               \
    const cutensorStatus_t katal_status_ = (expr);                   \
    if (katal_status_ != CUTENSOR_STATUS_SUCCESS)                    \
      throw ::katal::Error(::katal::Errc::CutensorError, (equation), \
                           cutensorGetErrorString(katal_status_));   \
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
