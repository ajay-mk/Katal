#ifndef KATAL_SRC_CUTENSOR_TYPES_HPP_
#define KATAL_SRC_CUTENSOR_TYPES_HPP_

#include <cutensor.h>

#include <katal/error.hpp>
#include <katal/options.hpp>
#include <katal/tensor_ref.hpp>

namespace katal::util {

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

inline cutensorAlgo_t to_cutensor(Algo algo) {
  switch (algo) {
    case Algo::Default:
      return CUTENSOR_ALGO_DEFAULT;
    case Algo::DefaultPatient:
      return CUTENSOR_ALGO_DEFAULT_PATIENT;
    case Algo::Gett:
      return CUTENSOR_ALGO_GETT;
    case Algo::Tgett:
      return CUTENSOR_ALGO_TGETT;
    case Algo::Ttgt:
      return CUTENSOR_ALGO_TTGT;
  }
  throw Error(Errc::UnsupportedPattern, "", "unrecognized algorithm");
}

inline cutensorWorksizePreference_t to_cutensor(
    WorkspacePreference preference) {
  switch (preference) {
    case WorkspacePreference::Min:
      return CUTENSOR_WORKSPACE_MIN;
    case WorkspacePreference::Default:
      return CUTENSOR_WORKSPACE_DEFAULT;
    case WorkspacePreference::Max:
      return CUTENSOR_WORKSPACE_MAX;
  }
  throw Error(Errc::UnsupportedPattern, "",
              "unrecognized workspace preference");
}

}  // namespace katal::util

#endif  // KATAL_SRC_CUTENSOR_TYPES_HPP_
