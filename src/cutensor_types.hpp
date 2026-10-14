#ifndef KATAL_SRC_CUTENSOR_TYPES_HPP_
#define KATAL_SRC_CUTENSOR_TYPES_HPP_

#include <string>
#include <string_view>

#include <cutensor.h>
#include <enumstr.hpp>

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

/// The storage type for a cuTENSOR data type, which is how the scalar type an
/// operation expects comes back.
inline Dtype from_cutensor(cutensorDataType_t type) {
  switch (type) {
    case CUTENSOR_R_32F:
      return Dtype::F32;
    case CUTENSOR_R_64F:
      return Dtype::F64;
    case CUTENSOR_C_32F:
      return Dtype::C32;
    case CUTENSOR_C_64F:
      return Dtype::C64;
    case CUTENSOR_R_16F:
      return Dtype::F16;
    case CUTENSOR_R_16BF:
      return Dtype::Bf16;
    default:
      throw Error(Errc::DtypeMismatch, "",
                  "cuTENSOR uses a scalar type Katal does not represent");
  }
}

/// The cuTENSOR compute descriptor for an accumulation type. cuTENSOR has no
/// complex compute descriptors: a complex contraction accumulates in the real
/// descriptor of the matching precision.
inline cutensorComputeDescriptor_t to_compute(Dtype type) {
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

/// The descriptor for a tensor-core mode, or for `compute` when Native.
/// `equation` is quoted by any error, as every backend error must.
inline cutensorComputeDescriptor_t to_compute(Dtype compute,
                                              Precision precision,
                                              std::string_view equation) {
  switch (precision) {
    case Precision::Native:
      return to_compute(compute);
    case Precision::Tf32:
      return CUTENSOR_COMPUTE_DESC_TF32;
    case Precision::Tf32x3:
      return CUTENSOR_COMPUTE_DESC_3XTF32;
    // Each emulated mode is gated at the release that introduced it.
    case Precision::Bf16x9:
#if CUTENSOR_VERSION >= 20400
      return CUTENSOR_COMPUTE_DESC_9X16BF;
#else
      throw Error(Errc::UnsupportedPattern, equation,
                  "precision " + enumstr::to_string(precision) +
                      " needs cuTENSOR 2.4 or newer");
#endif
    case Precision::F16x4:
#if CUTENSOR_VERSION >= 20500
      return CUTENSOR_COMPUTE_DESC_4X16F;
#else
      throw Error(Errc::UnsupportedPattern, equation,
                  "precision " + enumstr::to_string(precision) +
                      " needs cuTENSOR 2.5 or newer");
#endif
    case Precision::Int8x8:
#if CUTENSOR_VERSION >= 20400
      return CUTENSOR_COMPUTE_DESC_8XINT8;
#else
      throw Error(Errc::UnsupportedPattern, equation,
                  "precision " + enumstr::to_string(precision) +
                      " needs cuTENSOR 2.4 or newer");
#endif
  }
  throw Error(Errc::DtypeMismatch, equation, "unrecognized precision");
}

/// The oldest compute capability, as major * 10 + minor, that cuTENSOR's
/// contraction table lists for a tensor-core mode; 0 for Native. Checked
/// before planning, because cuTENSOR may quietly run an emulated mode in
/// native precision on an older GPU rather than refuse it.
inline int min_compute_capability(Precision precision) {
  switch (precision) {
    case Precision::Native:
      return 0;
    case Precision::Tf32:
    case Precision::Tf32x3:
      return 80;  // Ampere
    case Precision::Bf16x9:
    case Precision::Int8x8:
      return 90;  // Hopper
    case Precision::F16x4:
      return 100;  // Blackwell
  }
  return 0;
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
