#ifndef CUEIN_SCALAR_HPP_
#define CUEIN_SCALAR_HPP_

#include <complex>

#include <cuein/error.hpp>
#include <cuein/tensor_ref.hpp>

namespace cuein {

/// An alpha or beta value, kept type-erased so the ABI does not change with the
/// storage type and so the compute type may differ from it. Everything is held
/// widened to complex<double>, which loses nothing for the four types that
/// convert into it.
class Scalar {
 public:
  Scalar() = default;
  Scalar(float v) : value_(v), type_(Dtype::F32) {}                 // NOLINT
  Scalar(double v) : value_(v), type_(Dtype::F64) {}                // NOLINT
  Scalar(std::complex<float> v) : value_(v), type_(Dtype::C32) {}   // NOLINT
  Scalar(std::complex<double> v) : value_(v), type_(Dtype::C64) {}  // NOLINT

  Dtype type() const noexcept { return type_; }
  std::complex<double> value() const noexcept { return value_; }

  /// Narrows into `out`, which must have room for one element of `as`. This is
  /// what produces the pointer cuTENSOR wants for alpha and beta.
  void write_as(Dtype as, void* out) const {
    switch (as) {
      case Dtype::F32:
        *static_cast<float*>(out) = static_cast<float>(value_.real());
        return;
      case Dtype::F64:
        *static_cast<double*>(out) = value_.real();
        return;
      case Dtype::C32:
        *static_cast<std::complex<float>*>(out) = {
            static_cast<float>(value_.real()),
            static_cast<float>(value_.imag())};
        return;
      case Dtype::C64:
        *static_cast<std::complex<double>*>(out) = value_;
        return;
      case Dtype::F16:
      case Dtype::Bf16:
        throw Error(Errc::DtypeMismatch, "",
                    "alpha and beta cannot be expressed in half precision");
    }
    throw Error(Errc::DtypeMismatch, "", "unrecognized scalar type");
  }

 private:
  std::complex<double> value_{};
  Dtype type_ = Dtype::F64;
};

}  // namespace cuein

#endif  // CUEIN_SCALAR_HPP_
