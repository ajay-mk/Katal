#ifndef CUEIN_SRC_CUTENSOR_TYPES_HPP_
#define CUEIN_SRC_CUTENSOR_TYPES_HPP_

#include <cutensor.h>

#include <cuein/error.hpp>
#include <cuein/tensor_ref.hpp>

namespace cuein::util {

/// The cuTENSOR spelling of a storage type.
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

/// The cuTENSOR compute descriptor for an accumulation type. cuTENSOR has no
/// complex compute descriptors: a complex contraction accumulates in the real
/// descriptor of the matching precision.
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

}  // namespace cuein::util

#endif  // CUEIN_SRC_CUTENSOR_TYPES_HPP_
