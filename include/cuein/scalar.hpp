#ifndef CUEIN_SCALAR_HPP_
#define CUEIN_SCALAR_HPP_

#include <array>
#include <complex>
#include <cstddef>
#include <cstring>

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

 private:
  std::complex<double> value_{};
  Dtype type_ = Dtype::F64;
};

namespace util {

/// One scalar narrowed into the type an operation accumulates in, held inline
/// so that producing the pointer a C API wants costs no allocation. The
/// storage is sixteen bytes because that is the widest supported scalar.
class ScalarBytes {
 public:
  ScalarBytes(Scalar value, Dtype as) {
    const std::complex<double> v = value.value();
    switch (as) {
      case Dtype::F32:
        store(static_cast<float>(v.real()));
        return;
      case Dtype::F64:
        store(v.real());
        return;
      case Dtype::C32:
        store(std::complex<float>(static_cast<float>(v.real()),
                                  static_cast<float>(v.imag())));
        return;
      case Dtype::C64:
        store(v);
        return;
      case Dtype::F16:
      case Dtype::Bf16:
        throw Error(Errc::DtypeMismatch, "",
                    "alpha and beta cannot be expressed in half precision");
    }
    throw Error(Errc::DtypeMismatch, "", "unrecognized scalar type");
  }

  /// Borrowed from this object, so it must outlive the call it is passed to.
  const void* data() const noexcept { return storage_.data(); }

 private:
  template <class T>
  void store(T value) {
    static_assert(sizeof(T) <= sizeof(storage_));
    // memcpy rather than a cast: it starts the object's lifetime correctly and
    // compiles to the same single store.
    std::memcpy(storage_.data(), &value, sizeof(value));
  }

  alignas(16) std::array<std::byte, 16> storage_{};
};

}  // namespace util

}  // namespace cuein

#endif  // CUEIN_SCALAR_HPP_
