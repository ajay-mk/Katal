#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>

#include <katal/tensor_ref.hpp>

using katal::dense_strides;
using katal::Dtype;
using katal::Errc;
using katal::Extents;
using katal::Layout;
using katal::mode_order;
using katal::TensorRef;
using katal::test::thrown_code;
using katal::util::Index;

namespace {

// Asymmetric on purpose: a layout bug is invisible when every extent matches,
// and equal extents are what one writes by reflex.
constexpr std::array<std::int64_t, 4> shape{2, 3, 4, 5};

std::span<const std::int64_t> span_of(const std::array<std::int64_t, 4>& a) {
  return {a.data(), a.size()};
}

}  // namespace

TEST_CASE("dtype_of", "[tensor_ref]") {
  STATIC_REQUIRE(katal::dtype_of<float> == Dtype::F32);
  STATIC_REQUIRE(katal::dtype_of<double> == Dtype::F64);
  STATIC_REQUIRE(katal::dtype_of<std::complex<float>> == Dtype::C32);
  STATIC_REQUIRE(katal::dtype_of<std::complex<double>> == Dtype::C64);
}

TEST_CASE("dense strides", "[tensor_ref]") {
  SECTION("row major: the last mode varies fastest") {
    REQUIRE(dense_strides(span_of(shape), Layout::RowMajor) ==
            Extents{60, 20, 5, 1});
  }

  SECTION("column major: the first mode varies fastest") {
    REQUIRE(dense_strides(span_of(shape), Layout::ColumnMajor) ==
            Extents{1, 2, 6, 24});
  }

  SECTION("degenerate ranks") {
    const std::array<std::int64_t, 1> one{7};
    REQUIRE(dense_strides({one.data(), 1}, Layout::RowMajor) == Extents{1});
    REQUIRE(dense_strides({}, Layout::RowMajor) == Extents{});
  }
}

TEST_CASE("dense strides reject invalid or unrepresentable shapes",
          "[tensor_ref]") {
  const auto largest = std::numeric_limits<std::int64_t>::max();
  const Extents overflow{largest, 2, 1};
  REQUIRE(thrown_code([&] { dense_strides(overflow, Layout::ColumnMajor); }) ==
          Errc::ExtentMismatch);
  const Extents reversed{1, 2, largest};
  REQUIRE(thrown_code([&] { dense_strides(reversed, Layout::RowMajor); }) ==
          Errc::ExtentMismatch);
  const Extents negative{0, -1};
  REQUIRE(thrown_code([&] { dense_strides(negative, Layout::ColumnMajor); }) ==
          Errc::ExtentMismatch);
  const Extents zero{3, 0, 4};
  REQUIRE(thrown_code([&] { dense_strides(zero, Layout::RowMajor); }) ==
          Errc::ExtentMismatch);
  const Extents boundary{largest, 1};
  REQUIRE(dense_strides(boundary, Layout::ColumnMajor) == Extents{1, largest});
}

TEST_CASE("mode_order", "[tensor_ref]") {
  int storage = 0;

  SECTION("a row-major tensor is reversed into descriptor order") {
    const TensorRef t =
        katal::make_ref(&storage, Dtype::F64, span_of(shape), Layout::RowMajor);
    const auto m = mode_order(Index("ijkl"), t);

    REQUIRE(m.modes == Index("lkji"));
    REQUIRE(m.extent == Extents{5, 4, 3, 2});
    REQUIRE(m.stride == Extents{1, 5, 20, 60});
  }

  SECTION("reversal describes the same memory as the column-major spelling") {
    // The row-major tensor "ijkl" over 2,3,4,5 and the column-major tensor
    // "lkji" over 5,4,3,2 are the same bytes; descriptor order must agree.
    const std::array<std::int64_t, 4> reversed{5, 4, 3, 2};
    const TensorRef row =
        katal::make_ref(&storage, Dtype::F64, span_of(shape), Layout::RowMajor);
    const TensorRef col = katal::make_ref(
        &storage, Dtype::F64, span_of(reversed), Layout::ColumnMajor);

    const auto a = mode_order(Index("ijkl"), row);
    const auto b = mode_order(Index("lkji"), col);
    REQUIRE(a.modes == b.modes);
    REQUIRE(a.extent == b.extent);
    REQUIRE(a.stride == b.stride);
  }

  SECTION("a column-major tensor passes through untouched") {
    const TensorRef t = katal::make_ref(&storage, Dtype::F64, span_of(shape),
                                        Layout::ColumnMajor);
    const auto m = mode_order(Index("ijkl"), t);
    REQUIRE(m.modes == Index("ijkl"));
    REQUIRE(m.extent == Extents{2, 3, 4, 5});
    REQUIRE(m.stride == Extents{1, 2, 6, 24});
  }

  SECTION("explicit strides are carried through, reversed with the labels") {
    const std::array<std::int64_t, 4> padded{120, 20, 5, 1};
    TensorRef t =
        katal::make_ref(&storage, Dtype::F64, span_of(shape), Layout::RowMajor);
    t.stride = {padded.data(), padded.size()};

    const auto m = mode_order(Index("ijkl"), t);
    REQUIRE(m.modes == Index("lkji"));
    REQUIRE(m.stride == Extents{1, 5, 20, 120});
  }

  SECTION("rank disagreements are rejected") {
    const TensorRef t =
        katal::make_ref(&storage, Dtype::F64, span_of(shape), Layout::RowMajor);
    REQUIRE_THROWS_AS(mode_order(Index("ijk"), t), katal::Error);

    const std::array<std::int64_t, 4> two{1, 1, 1, 1};
    TensorRef bad = t;
    bad.stride = {two.data(), 2};
    REQUIRE_THROWS_AS(mode_order(Index("ijkl"), bad), katal::Error);
  }

  SECTION("a rank error quotes the equation and names the operand") {
    const TensorRef t =
        katal::make_ref(&storage, Dtype::F64, span_of(shape), Layout::RowMajor);
    try {
      mode_order(Index("ijk"), t, "ijk,kl->ijl");
      FAIL("expected a rejection");
    } catch (const katal::Error& e) {
      REQUIRE(e.code() == Errc::RankMismatch);
      REQUIRE(e.equation() == "ijk,kl->ijl");
      REQUIRE(std::string(e.what()).find("tensor \"ijk\" has 3 labels") !=
              std::string::npos);
    }
  }
}

TEST_CASE("make_ref derives the alignment from the address", "[tensor_ref]") {
  alignas(256) std::array<double, 8> buffer{};
  const auto alignment = [&](std::size_t offset_bytes) {
    return katal::make_ref(
               reinterpret_cast<std::byte*>(buffer.data()) + offset_bytes,
               Dtype::F64, span_of(shape))
        .alignment;
  };
  REQUIRE(alignment(0) == 256);
  // A sub-block of a larger allocation, the usual case in coupled-cluster
  // codes, used to be asserted 256-byte aligned too.
  REQUIRE(alignment(sizeof(double)) == 8);
  REQUIRE(alignment(2 * sizeof(double)) == 16);
  REQUIRE(alignment(1) == 1);
  REQUIRE(alignment(512) == 256);
  REQUIRE(katal::make_ref(nullptr, Dtype::F64, span_of(shape)).alignment ==
          256);
}

TEST_CASE("inputs are const-correct", "[tensor_ref]") {
  std::array<double, 120> storage{};
  const std::array<double, 120>& read_only = storage;

  STATIC_REQUIRE(std::is_same_v<decltype(katal::make_ref(read_only.data(),
                                                         Dtype::F64, shape)),
                                katal::ConstTensorRef>);
  STATIC_REQUIRE(std::is_same_v<decltype(katal::make_ref(storage.data(),
                                                         Dtype::F64, shape)),
                                TensorRef>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(katal::make_ref(nullptr, Dtype::F64, shape)),
                     TensorRef>);
  // Writable converts to read-only, never the reverse.
  STATIC_REQUIRE(std::is_convertible_v<TensorRef, katal::ConstTensorRef>);
  STATIC_REQUIRE_FALSE(std::is_convertible_v<katal::ConstTensorRef, TensorRef>);

  TensorRef writable =
      katal::make_ref(storage.data(), Dtype::F64, span_of(shape));
  const std::array<std::int64_t, 4> stride{60, 20, 5, 1};
  writable.stride = {stride.data(), stride.size()};
  writable.layout = Layout::RowMajor;
  const katal::ConstTensorRef read = writable;
  REQUIRE(read.data == writable.data);
  REQUIRE(read.type == writable.type);
  REQUIRE(read.extent.data() == writable.extent.data());
  REQUIRE(read.stride.data() == writable.stride.data());
  REQUIRE(read.layout == writable.layout);
  REQUIRE(read.alignment == writable.alignment);
  REQUIRE(mode_order(Index("ijkl"), read).modes == Index("lkji"));
}
