// cuTENSOR against the CPU reference backend: each case runs the same call
// on the device, through the default context, and on the host, through the
// CPU backend, from the same data, and the results must agree. Sums of
// products are checked against reference_einsum in test_gpu.cpp; this covers
// what that reference cannot compute: other reductions, elementwise
// operators, permutations and unary operators, in every tested type, and
// three-operand contractions, which cuTENSOR runs as one trinary kernel.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

#include "cpu_backend.hpp"
#include "cuda_memory.hpp"
#include "enum_names.hpp"
#include "numeric.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/context.hpp>
#include <katal/plan.hpp>
#include <katal/tensor_ref.hpp>

namespace {

using namespace katal;
using katal::test::count;
using katal::test::random_values;

using Shape = std::vector<std::int64_t>;

bool have_gpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::vector<T>& host) : size_(host.size()) {
    T* data = nullptr;
    REQUIRE(cudaMalloc(&data, bytes()) == cudaSuccess);
    ptr_.reset(data);
    REQUIRE(cudaMemcpy(ptr_.get(), host.data(), bytes(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const noexcept { return ptr_.get(); }

  std::vector<T> to_host() const {
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<T> host(size_);
    REQUIRE(cudaMemcpy(host.data(), ptr_.get(), bytes(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    return host;
  }

 private:
  std::size_t bytes() const noexcept { return size_ * sizeof(T); }

  std::unique_ptr<T, katal::test::DeviceFree> ptr_;
  std::size_t size_ = 0;
};

/// One input: its shape, how it is read, and the range its values come from.
struct Input {
  Shape shape;
  UnaryOp op = UnaryOp::Identity;
  double lo = -1.0;
  double hi = 1.0;
};

/// Runs `call(context, allocator, inputs, d)` once with cuTENSOR and once
/// with the CPU backend, from the same inputs and the same starting d, and
/// checks that the two results agree. `call` is generic over the allocator,
/// which is device memory for one and host memory for the other.
template <class T, class Call>
void compare(const std::vector<Input>& inputs, const Shape& out, Call call) {
  std::vector<std::vector<T>> host;
  for (std::size_t k = 0; k < inputs.size(); ++k)
    host.push_back(random_values<T>(count(inputs[k].shape),
                                    static_cast<std::uint32_t>(k + 1),
                                    inputs[k].lo, inputs[k].hi));
  const std::vector<T> start = random_values<T>(count(out), 99);

  const auto refs = [&](const auto& pointer) {
    std::vector<ConstTensorRef> result;
    for (std::size_t k = 0; k < inputs.size(); ++k)
      result.push_back(
          apply(inputs[k].op, make_ref(static_cast<const void*>(pointer(k)),
                                       dtype_of<T>, inputs[k].shape)));
    return result;
  };

  // The host run.
  std::vector<T> cpu = start;
  const Context host_context = katal::test::cpu_context();
  call(host_context, katal::test::HostAllocator{},
       refs([&](std::size_t k) { return host[k].data(); }),
       make_ref(static_cast<void*>(cpu.data()), dtype_of<T>, out));

  // The device run.
  std::vector<std::unique_ptr<DeviceBuffer<T>>> device;
  for (const std::vector<T>& h : host)
    device.push_back(std::make_unique<DeviceBuffer<T>>(h));
  const DeviceBuffer<T> dd(start);
  call(default_context(), CudaAsyncAllocator{},
       refs([&](std::size_t k) { return device[k]->get(); }),
       make_ref(static_cast<void*>(dd.get()), dtype_of<T>, out));
  const std::vector<T> gpu = dd.to_host();

  // cuTENSOR may block and order its sums differently, so allow ten times
  // the host tolerance.
  CHECK(katal::test::close(gpu, cpu, 10 * katal::test::tolerance<T>()));
}

}  // namespace

TEMPLATE_TEST_CASE("cuTENSOR and the CPU backend agree on reductions",
                   "[cpu_backend][gpu]", float, double) {
  if (!have_gpu()) SKIP("no CUDA device");
  using T = TestType;
  for (const BinaryOp op : {BinaryOp::Max, BinaryOp::Min, BinaryOp::Multiply}) {
    CAPTURE(op);
    // Factors near one, so that a product of eight stays well scaled.
    const double lo = op == BinaryOp::Multiply ? 0.5 : -1.0;
    compare<T>({{{6, 8}, UnaryOp::Identity, lo, 1.5}}, {6},
               [&](const Context& context, auto allocator,
                   const std::vector<ConstTensorRef>& in, const TensorRef& d) {
                 reduce("ij->i", op, T(2), in[0], T(0.5), d, {}, {}, context,
                        allocator);
               });
  }
}

TEMPLATE_TEST_CASE("cuTENSOR and the CPU backend agree elementwise",
                   "[cpu_backend][gpu]", float, double, std::complex<float>,
                   std::complex<double>) {
  if (!have_gpu()) SKIP("no CUDA device");
  using T = TestType;
  std::vector<BinaryOp> ops{BinaryOp::Add, BinaryOp::Multiply};
  if constexpr (std::is_floating_point_v<T>) {
    ops.push_back(BinaryOp::Max);
    ops.push_back(BinaryOp::Min);
  }
  for (const BinaryOp op : ops) {
    CAPTURE(op);
    // b broadcast along i, and a transposed against d.
    compare<T>({{{9, 7}}, {{9}}}, {7, 9},
               [&](const Context& context, auto allocator,
                   const std::vector<ConstTensorRef>& in, const TensorRef& d) {
                 elementwise("ji,j->ij", op, T(2), in[0], T(-1), in[1], T(0.5),
                             d, {}, {}, context, allocator);
               });
  }
}

TEMPLATE_TEST_CASE("cuTENSOR and the CPU backend agree on permutations",
                   "[cpu_backend][gpu]", float, double, std::complex<float>,
                   std::complex<double>) {
  if (!have_gpu()) SKIP("no CUDA device");
  using T = TestType;
  for (const char* equation : {"ijk->kij", "i->ij"}) {
    CAPTURE(equation);
    const bool broadcast = equation[1] == '-';
    compare<T>({{broadcast ? Shape{11} : Shape{4, 5, 6}}},
               broadcast ? Shape{11, 3} : Shape{6, 4, 5},
               [&](const Context& context, auto allocator,
                   const std::vector<ConstTensorRef>& in, const TensorRef& d) {
                 permute(equation, T(3), in[0], d, {}, {}, context, allocator);
               });
  }
}

TEMPLATE_TEST_CASE("cuTENSOR and the CPU backend agree on unary operators",
                   "[cpu_backend][gpu]", float, double, std::complex<float>,
                   std::complex<double>) {
  if (!have_gpu()) SKIP("no CUDA device");
  using T = TestType;
  constexpr bool complex = !std::is_floating_point_v<T>;

  struct Case {
    UnaryOp op;
    // The range the input is drawn from, inside the operator's domain.
    double lo;
    double hi;
    // Defined only on real values.
    bool real_only = false;
  };
  const std::vector<Case> cases{
      {UnaryOp::Conj, -1, 1},
      {UnaryOp::Neg, -1, 1},
      {UnaryOp::Abs, -1, 1},
      {UnaryOp::Sqrt, 0.1, 2},
      {UnaryOp::Rcp, 0.5, 2},
      {UnaryOp::Exp, -1, 1},
      {UnaryOp::Log, 0.1, 2},
      {UnaryOp::Sin, -1, 1},
      {UnaryOp::Cos, -1, 1},
      {UnaryOp::Tan, -1, 1},
      {UnaryOp::Sinh, -1, 1},
      {UnaryOp::Cosh, -1, 1},
      {UnaryOp::Tanh, -1, 1},
      {UnaryOp::Asin, -0.9, 0.9},
      {UnaryOp::Acos, -0.9, 0.9},
      {UnaryOp::Atan, -1, 1},
      {UnaryOp::Asinh, -1, 1},
      {UnaryOp::Acosh, 1.1, 2},
      {UnaryOp::Atanh, -0.9, 0.9},
      {UnaryOp::Ceil, -2, 2, true},
      {UnaryOp::Floor, -2, 2, true},
      {UnaryOp::Relu, -1, 1, true},
      {UnaryOp::Sigmoid, -2, 2, true},
      {UnaryOp::Mish, -2, 2, true},
      {UnaryOp::Swish, -2, 2, true},
      {UnaryOp::SoftPlus, -2, 2, true},
      {UnaryOp::SoftSign, -2, 2, true},
  };
  // cuTENSOR's elementwise kernels accept fewer operators than the CPU
  // backend: no Conj on real data, and only Conj on complex data. Pinned, so
  // that a cuTENSOR that starts accepting one gets compared.
  std::vector<UnaryOp> refused;
  std::vector<UnaryOp> expected;
  for (const Case& c : cases) {
    if (complex && c.real_only) continue;
    if (complex ? c.op != UnaryOp::Conj : c.op == UnaryOp::Conj)
      expected.push_back(c.op);
    CAPTURE(c.op);
    try {
      compare<T>(
          {{{5, 6}, c.op, c.lo, c.hi}, {{5, 6}}}, {5, 6},
          [&](const Context& context, auto allocator,
              const std::vector<ConstTensorRef>& in, const TensorRef& d) {
            add("ij,ij->ij", T(1), in[0], T(1), in[1], d, {}, {}, context,
                allocator);
          });
    } catch (const Error&) {
      // The CPU backend runs first and accepts every operator here, so this
      // is cuTENSOR refusing it.
      refused.push_back(c.op);
    }
  }
  CHECK(refused == expected);
}

TEMPLATE_TEST_CASE(
    "cuTENSOR and the CPU backend agree on three-operand "
    "contractions",
    "[cpu_backend][gpu][trinary]", float, double, std::complex<float>,
    std::complex<double>) {
  if (!have_gpu()) SKIP("no CUDA device");
  using T = TestType;
  // A chain with c's own label y summed out first, and a coupled-cluster
  // shape with a permuted result.
  const struct {
    const char* equation;
    std::vector<Input> inputs;
    Shape out;
  } cases[] = {
      {"ij,jk,kly->li", {{{6, 7}}, {{7, 8}}, {{8, 5, 3}}}, {5, 6}},
      {"ijab,abmn,mnkl->klij",
       {{{3, 4, 5, 2}}, {{5, 2, 4, 3}}, {{4, 3, 2, 5}}},
       {2, 5, 3, 4}},
  };
  for (const auto& c : cases) {
    CAPTURE(c.equation);
    try {
      compare<T>(
          c.inputs, c.out,
          [&](const Context& context, auto allocator,
              const std::vector<ConstTensorRef>& in, const TensorRef& d) {
            einsum(c.equation, T(1.5), in[0], in[1], in[2], T(-0.5), d, {}, {},
                   context, allocator);
          });
    } catch (const Error& e) {
      // The CPU backend runs first and has no version limit, so this is a
      // cuTENSOR older than 2.2.
      if (std::string_view(e.what()).find("cuTENSOR 2.2") ==
          std::string_view::npos)
        throw;
      SKIP("cuTENSOR older than 2.2 has no trinary contraction");
    }
  }
}
