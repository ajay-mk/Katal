// What the front end hands a backend, checked on the host: a recording
// backend stands in for cuTENSOR, so no device is needed.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "backend.hpp"
#include "context_impl.hpp"
#include <catch2/catch_test_macros.hpp>

#include <katal/context.hpp>
#include <katal/plan.hpp>
#include <katal/tensor_ref.hpp>

namespace {

using katal::backend::Operand;
using katal::backend::Operation;
using katal::backend::OpKind;

struct Run {
  katal::Scalar alpha;
  std::vector<const void*> inputs;
  katal::Scalar beta;
  const void* c = nullptr;
  void* d = nullptr;
  katal::Workspace workspace;
};

/// What the backend saw, shared with the test after the context owns it.
struct Record {
  std::optional<Operation> operation;
  std::uint64_t workspace = 0;
  std::vector<Run> runs;
};

class RecordingKernel : public katal::backend::Kernel {
 public:
  RecordingKernel(std::shared_ptr<Record> record, std::size_t n_inputs)
      : record_(std::move(record)), n_inputs_(n_inputs) {}

  std::uint64_t workspace_size() const noexcept override {
    return record_->workspace;
  }

  void run(katal::Scalar alpha, const void* const* inputs, katal::Scalar beta,
           const void* c, void* d, katal::Workspace workspace,
           katal::StreamRef) const override {
    // Parentheses, not braces: braces would store the two bounds themselves.
    std::vector<const void*> pointers(inputs, inputs + n_inputs_);
    record_->runs.push_back(
        {alpha, std::move(pointers), beta, c, d, workspace});
  }

 private:
  std::shared_ptr<Record> record_;
  std::size_t n_inputs_;
};

class RecordingBackend : public katal::backend::Backend {
 public:
  explicit RecordingBackend(std::shared_ptr<Record> record)
      : record_(std::move(record)) {}

  int device() const noexcept override { return 0; }
  void check_device(std::string_view, std::string_view) const override {}

  std::unique_ptr<katal::backend::Kernel> plan(
      const Operation& operation, const katal::PlanOptions&) const override {
    record_->operation = operation;
    return std::make_unique<RecordingKernel>(record_, operation.inputs.size());
  }

 private:
  std::shared_ptr<Record> record_;
};

std::vector<std::int32_t> ids(const char* labels) {
  std::vector<std::int32_t> result;
  for (; *labels != '\0'; ++labels)
    result.push_back(static_cast<unsigned char>(*labels));
  return result;
}

std::vector<std::int64_t> list(const katal::Extents& values) {
  return {values.begin(), values.end()};
}

alignas(256) std::array<double, 64> buffer_a{};
alignas(256) std::array<double, 64> buffer_b{};
alignas(256) std::array<double, 64> buffer_c{};
alignas(256) std::array<double, 64> buffer_d{};
alignas(256) std::array<unsigned char, 1024> scratch{};

}  // namespace

TEST_CASE("a contraction lowers to an operation in descriptor order",
          "[backend]") {
  auto record = std::make_shared<Record>();
  const auto context = katal::Context::Impl::with_backend(
      std::make_unique<RecordingBackend>(record));
  const std::array<std::int64_t, 2> ea{2, 3}, eb{3, 4}, ed{2, 4};
  const auto f64 = katal::Dtype::F64;

  SECTION("row-major operands have their modes reversed") {
    katal::make_plan(context, katal::Equation::parse("ij,jk->ik"),
                     katal::make_ref(nullptr, f64, ea),
                     katal::make_ref(nullptr, f64, eb),
                     katal::make_ref(nullptr, f64, ed));
    REQUIRE(record->operation);
    const Operation& op = *record->operation;
    CHECK(op.kind == OpKind::Contraction);
    CHECK(op.equation == "ij,jk->ik");
    CHECK(op.compute == f64);
    REQUIRE(op.inputs.size() == 2);
    CHECK(op.inputs[0].modes == ids("ji"));
    CHECK(list(op.inputs[0].extent) == std::vector<std::int64_t>{3, 2});
    CHECK(list(op.inputs[0].stride) == std::vector<std::int64_t>{1, 3});
    CHECK(op.inputs[1].modes == ids("kj"));
    CHECK(list(op.inputs[1].stride) == std::vector<std::int64_t>{1, 4});
    CHECK(op.output.modes == ids("ki"));
    CHECK(list(op.output.extent) == std::vector<std::int64_t>{4, 2});
    CHECK(op.output.alignment == 256);
  }

  SECTION("column-major operands keep the equation's order") {
    katal::make_plan(
        context, katal::Equation::parse("ij,jk->ik"),
        katal::make_ref(nullptr, f64, ea, katal::Layout::ColumnMajor),
        katal::make_ref(nullptr, f64, eb, katal::Layout::ColumnMajor),
        katal::make_ref(nullptr, f64, ed, katal::Layout::ColumnMajor));
    REQUIRE(record->operation);
    CHECK(record->operation->inputs[0].modes == ids("ij"));
    CHECK(list(record->operation->inputs[0].stride) ==
          std::vector<std::int64_t>{1, 2});
    CHECK(record->operation->output.modes == ids("ik"));
  }

  SECTION("each operand carries its own type and alignment") {
    auto a = katal::make_ref(nullptr, katal::Dtype::C64, ea);
    a.alignment = 16;
    katal::make_plan(context, katal::Equation::parse("ij,jk->ik"), a,
                     katal::make_ref(nullptr, f64, eb),
                     katal::make_ref(nullptr, katal::Dtype::C64, ed));
    REQUIRE(record->operation);
    CHECK(record->operation->inputs[0].type == katal::Dtype::C64);
    CHECK(record->operation->inputs[0].alignment == 16);
    CHECK(record->operation->inputs[1].type == f64);
    CHECK(record->operation->compute == katal::Dtype::C64);
  }
}

TEST_CASE("the compute type defaults to the output's", "[backend]") {
  auto record = std::make_shared<Record>();
  const auto context = katal::Context::Impl::with_backend(
      std::make_unique<RecordingBackend>(record));
  const std::array<std::int64_t, 1> e{4};
  const auto plan = [&](katal::Dtype type, katal::PlanOptions options = {}) {
    katal::make_plan(context, katal::Equation::parse("i->i"),
                     katal::make_ref(nullptr, type, e),
                     katal::make_ref(nullptr, type, e), options);
    REQUIRE(record->operation);
    return record->operation->compute;
  };

  CHECK(plan(katal::Dtype::F32) == katal::Dtype::F32);
  CHECK(plan(katal::Dtype::C32) == katal::Dtype::C32);
  CHECK(plan(katal::Dtype::F16) == katal::Dtype::F32);
  CHECK(plan(katal::Dtype::Bf16) == katal::Dtype::F32);
  katal::PlanOptions options;
  options.compute = katal::Dtype::F64;
  CHECK(plan(katal::Dtype::F32, options) == katal::Dtype::F64);
}

TEST_CASE("one operand lowers to a reduction", "[backend]") {
  auto record = std::make_shared<Record>();
  const auto context = katal::Context::Impl::with_backend(
      std::make_unique<RecordingBackend>(record));
  const std::array<std::int64_t, 2> ea{2, 3};
  const std::array<std::int64_t, 1> ed{2};
  katal::make_plan(context, katal::Equation::parse("ij->i"),
                   katal::make_ref(nullptr, katal::Dtype::F32, ea),
                   katal::make_ref(nullptr, katal::Dtype::F32, ed));
  REQUIRE(record->operation);
  CHECK(record->operation->kind == OpKind::Reduction);
  REQUIRE(record->operation->inputs.size() == 1);
  CHECK(record->operation->inputs[0].modes == ids("ji"));
  CHECK(record->operation->output.modes == ids("i"));
}

TEST_CASE("execute checks its arguments before the kernel runs", "[backend]") {
  auto record = std::make_shared<Record>();
  record->workspace = 512;
  const auto context = katal::Context::Impl::with_backend(
      std::make_unique<RecordingBackend>(record));
  const std::array<std::int64_t, 2> e{2, 2};
  const auto f64 = katal::Dtype::F64;
  const auto plan = katal::make_plan(
      context, katal::Equation::parse("ij,jk->ik"),
      katal::make_ref(nullptr, f64, e), katal::make_ref(nullptr, f64, e),
      katal::make_ref(nullptr, f64, e));
  CHECK(plan.workspace_size() == 512);
  const katal::Workspace enough{scratch.data(), scratch.size()};

  SECTION("inputs, c and d reach the kernel in order") {
    plan.execute(2.0, buffer_a.data(), buffer_b.data(), 0.5, buffer_c.data(),
                 buffer_d.data(), enough);
    REQUIRE(record->runs.size() == 1);
    const Run& run = record->runs[0];
    CHECK(run.alpha.value() == 2.0);
    CHECK(run.beta.value() == 0.5);
    CHECK(run.inputs ==
          std::vector<const void*>{buffer_a.data(), buffer_b.data()});
    CHECK(run.c == buffer_c.data());
    CHECK(run.d == buffer_d.data());
    CHECK(run.workspace.ptr == scratch.data());
    CHECK(run.workspace.size == scratch.size());
  }

  SECTION("a small workspace is refused") {
    CHECK_THROWS_AS(
        plan.execute(1.0, buffer_a.data(), buffer_b.data(), 0.0,
                     buffer_d.data(), buffer_d.data(), {scratch.data(), 100}),
        katal::Error);
    CHECK(record->runs.empty());
  }

  SECTION("a pointer less aligned than planned is refused") {
    CHECK_THROWS_AS(plan.execute(1.0, buffer_a.data() + 1, buffer_b.data(), 0.0,
                                 buffer_d.data(), buffer_d.data(), enough),
                    katal::Error);
    CHECK(record->runs.empty());
  }

  SECTION("the one-operand execute is refused") {
    CHECK_THROWS_AS(plan.execute(1.0, buffer_a.data(), 0.0, buffer_d.data(),
                                 buffer_d.data(), enough),
                    katal::Error);
    CHECK(record->runs.empty());
  }
}
