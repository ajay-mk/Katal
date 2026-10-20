#ifndef KATAL_TEST_CPU_BACKEND_HPP_
#define KATAL_TEST_CPU_BACKEND_HPP_

#include <cstdint>
#include <memory>

#include "backend.hpp"

#include <katal/context.hpp>
#include <katal/plan.hpp>

/// A reference backend that runs every Operation with plain loops over host
/// memory. It exists so the tests can check numbers: through the whole
/// frontend without a GPU, and against cuTENSOR on one. It is written to be
/// obviously right rather than fast, and is not part of the library.
namespace katal::test {

/// Plans and runs on the host. Pointers handed to its kernels are host
/// memory, the stream is ignored and every run finishes before it returns.
///
/// It computes in the Operation's compute type and ignores the PlanOptions,
/// so a tensor-core precision runs natively. It accepts F32, F64, C32 and
/// C64, the tested types, and refuses F16 and Bf16. A complex compute type
/// refuses the unary operators with no complex meaning (Ceil, Floor, Relu,
/// Sigmoid, Mish, Swish, SoftPlus, SoftSign) and Max and Min.
std::unique_ptr<backend::Backend> make_cpu_backend();

/// A context that plans through make_cpu_backend.
Context cpu_context(Config config = {});

/// Host scratch for a plan, aligned as cuTENSOR's device allocations are, so
/// a plan that keeps reduced inputs in its workspace accepts it.
class HostAllocator {
 public:
  Workspace allocate(std::uint64_t bytes, StreamRef stream) const;
  void deallocate(Workspace workspace, StreamRef stream) const noexcept;
};

}  // namespace katal::test

#endif  // KATAL_TEST_CPU_BACKEND_HPP_
