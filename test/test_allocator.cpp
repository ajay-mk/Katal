#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <cuein/plan.hpp>

using cuein::CudaAsyncAllocator;
using cuein::StreamRef;
using cuein::Workspace;
using cuein::WorkspaceAllocator;

namespace {

/// What a caller with its own device pool would write: a cheap handle that
/// refers to the pool rather than owning it.
class PoolHandle {
 public:
  explicit PoolHandle(std::byte* pool) : pool_(pool) {}
  Workspace allocate(uint64_t bytes, StreamRef) { return {pool_, bytes}; }
  void deallocate(Workspace, StreamRef) {}

 private:
  std::byte* pool_ = nullptr;
};

struct WrongReturnType {
  void* allocate(uint64_t, StreamRef) { return nullptr; }
  void deallocate(Workspace, StreamRef) {}
};

struct MissingDeallocate {
  Workspace allocate(uint64_t, StreamRef) { return {}; }
};

}  // namespace

// The concept is the contract, so check it accepts and rejects the right
// shapes. None of this needs a GPU.
static_assert(WorkspaceAllocator<CudaAsyncAllocator>);
static_assert(WorkspaceAllocator<PoolHandle>);
static_assert(!WorkspaceAllocator<WrongReturnType>);
static_assert(!WorkspaceAllocator<MissingDeallocate>);
static_assert(!WorkspaceAllocator<int>);

TEST_CASE("a caller's allocator is used verbatim", "[allocator]") {
  std::byte storage[64]{};
  PoolHandle pool(storage);

  const Workspace got = pool.allocate(32, {});
  REQUIRE(got.ptr == storage);
  REQUIRE(got.size == 32);
}
