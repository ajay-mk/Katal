#ifndef DEVICEEIN_SRC_CUTENSOR_STATUS_HPP_
#define DEVICEEIN_SRC_CUTENSOR_STATUS_HPP_

#include <cuda_runtime.h>
#include <cutensor.h>

#include <deviceein/error.hpp>

/// Turns a cuTENSOR status into a deviceein::Error. The equation argument is
/// required rather than optional because it is what makes a cuTENSOR failure
/// diagnosable.
#define DEVICEEIN_CHECK(expr, equation)                                      \
  do {                                                                       \
    const cutensorStatus_t deviceein_status_ = (expr);                       \
    if (deviceein_status_ != CUTENSOR_STATUS_SUCCESS)                        \
      throw ::deviceein::Error(::deviceein::Errc::CutensorError, (equation), \
                               cutensorGetErrorString(deviceein_status_));   \
  } while (false)

/// The same for a CUDA runtime call.
#define DEVICEEIN_CHECK_CUDA(expr, equation)                             \
  do {                                                                   \
    const cudaError_t deviceein_cuda_ = (expr);                          \
    if (deviceein_cuda_ != cudaSuccess)                                  \
      throw ::deviceein::Error(::deviceein::Errc::CudaError, (equation), \
                               cudaGetErrorString(deviceein_cuda_));     \
  } while (false)

#endif  // DEVICEEIN_SRC_CUTENSOR_STATUS_HPP_
