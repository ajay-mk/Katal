#ifndef CUEIN_SRC_CUTENSOR_STATUS_HPP_
#define CUEIN_SRC_CUTENSOR_STATUS_HPP_

#include <cuda_runtime.h>
#include <cutensor.h>

#include <cuein/error.hpp>

/// Turns a cuTENSOR status into a cuein::Error. The equation argument is
/// required rather than optional because it is what makes a cuTENSOR failure
/// diagnosable.
#define CUEIN_CHECK(expr, equation)                                  \
  do {                                                               \
    const cutensorStatus_t cuein_status_ = (expr);                   \
    if (cuein_status_ != CUTENSOR_STATUS_SUCCESS)                    \
      throw ::cuein::Error(::cuein::Errc::CutensorError, (equation), \
                           cutensorGetErrorString(cuein_status_));   \
  } while (false)

/// The same for a CUDA runtime call.
#define CUEIN_CHECK_CUDA(expr, equation)                         \
  do {                                                           \
    const cudaError_t cuein_cuda_ = (expr);                      \
    if (cuein_cuda_ != cudaSuccess)                              \
      throw ::cuein::Error(::cuein::Errc::CudaError, (equation), \
                           cudaGetErrorString(cuein_cuda_));     \
  } while (false)

#endif  // CUEIN_SRC_CUTENSOR_STATUS_HPP_
