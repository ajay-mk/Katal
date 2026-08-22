#ifndef CUEIN_SRC_CUTENSOR_GLUE_HPP_
#define CUEIN_SRC_CUTENSOR_GLUE_HPP_

#include <cuda_runtime.h>
#include <cutensor.h>

#include <cuein/error.hpp>
#include <cuein/tensor_ref.hpp>

// Everything in this header exists to cross into cuTENSOR's C API: status
// codes become cuein::Error, and Dtype becomes cuTENSOR's enums. It is why
// <cutensor.h> stays out of the public headers.

/// The equation argument is what makes a cuTENSOR failure diagnosable, so it
/// is required rather than optional.
#define CUEIN_CHECK(expr, equation)                                  \
  do {                                                               \
    const cutensorStatus_t cuein_status_ = (expr);                   \
    if (cuein_status_ != CUTENSOR_STATUS_SUCCESS)                    \
      throw ::cuein::Error(::cuein::Errc::CutensorError, (equation), \
                           cutensorGetErrorString(cuein_status_));   \
  } while (false)

#define CUEIN_CHECK_CUDA(expr, equation)                         \
  do {                                                           \
    const cudaError_t cuein_cuda_ = (expr);                      \
    if (cuein_cuda_ != cudaSuccess)                              \
      throw ::cuein::Error(::cuein::Errc::CudaError, (equation), \
                           cudaGetErrorString(cuein_cuda_));     \
  } while (false)

namespace cuein::detail {

inline cutensorDataType_t to_cutensor(Dtype type) {
  switch (type) {
    case Dtype::F32:
      return CUTENSOR_R_32F;
    case Dtype::F64:
      return CUTENSOR_R_64F;
    case Dtype::C32:
      return CUTENSOR_C_32F;
    case Dtype::C64:
      return CUTENSOR_C_64F;
    case Dtype::F16:
      return CUTENSOR_R_16F;
    case Dtype::Bf16:
      return CUTENSOR_R_16BF;
  }
  throw Error(Errc::DtypeMismatch, "", "unrecognized data type");
}

/// cuTENSOR has no complex compute descriptors: a complex contraction uses the
/// real descriptor of the matching precision.
inline const cutensorComputeDescriptor_t to_compute(Dtype type) {
  switch (type) {
    case Dtype::F32:
    case Dtype::C32:
      return CUTENSOR_COMPUTE_DESC_32F;
    case Dtype::F64:
    case Dtype::C64:
      return CUTENSOR_COMPUTE_DESC_64F;
    case Dtype::F16:
      return CUTENSOR_COMPUTE_DESC_16F;
    case Dtype::Bf16:
      return CUTENSOR_COMPUTE_DESC_16BF;
  }
  throw Error(Errc::DtypeMismatch, "", "unrecognized compute type");
}

}  // namespace cuein::detail

#endif  // CUEIN_SRC_CUTENSOR_GLUE_HPP_
