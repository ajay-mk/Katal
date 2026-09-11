#include <complex>
#include <cstring>

#include <catch2/catch_test_macros.hpp>

#include <katal/scalar.hpp>

using katal::Dtype;
using katal::Scalar;
using katal::util::ScalarBytes;

namespace {

template <class T>
T read_back(const ScalarBytes& bytes) {
  T value{};
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

}  // namespace

TEST_CASE("Scalar remembers the type it was given", "[scalar]") {
  REQUIRE(Scalar(1.0F).type() == Dtype::F32);
  REQUIRE(Scalar(1.0).type() == Dtype::F64);
  REQUIRE(Scalar(std::complex<float>{1, 2}).type() == Dtype::C32);
  REQUIRE(Scalar(std::complex<double>{1, 2}).type() == Dtype::C64);
  REQUIRE(Scalar().value() == std::complex<double>{});
}

TEST_CASE("ScalarBytes narrows into the accumulation type", "[scalar]") {
  SECTION("a double is delivered as whatever the operation computes in") {
    REQUIRE(read_back<double>(ScalarBytes(2.5, Dtype::F64)) == 2.5);
    REQUIRE(read_back<float>(ScalarBytes(2.5, Dtype::F32)) == 2.5F);
    REQUIRE(read_back<std::complex<double>>(ScalarBytes(2.5, Dtype::C64)) ==
            std::complex<double>{2.5, 0.0});
  }

  SECTION("a complex value survives, and narrows") {
    const std::complex<double> v{1.5, -2.5};
    REQUIRE(read_back<std::complex<double>>(ScalarBytes(v, Dtype::C64)) == v);
    REQUIRE(read_back<std::complex<float>>(ScalarBytes(v, Dtype::C32)) ==
            std::complex<float>{1.5F, -2.5F});
  }

  SECTION("asking for a real type drops the imaginary part") {
    REQUIRE(read_back<double>(ScalarBytes(std::complex<double>{3.0, 9.0},
                                          Dtype::F64)) == 3.0);
  }

  SECTION("half precision has no scalar spelling") {
    REQUIRE_THROWS_AS(ScalarBytes(1.0, Dtype::F16), katal::Error);
    REQUIRE_THROWS_AS(ScalarBytes(1.0, Dtype::Bf16), katal::Error);
  }
}
