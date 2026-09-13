#include <cstdint>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include <katal/plan.hpp>

using katal::CudaAsyncAllocator;
using katal::StreamRef;
using katal::Workspace;
using katal::WorkspaceAllocator;

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

TEST_CASE("workspace allocation preserves the caller's stream and lifetime",
          "[allocator]") {
  struct State {
    std::byte storage[64]{};
    std::uint64_t requested = 0;
    StreamRef allocated_on;
    StreamRef released_on;
    Workspace released;
    int allocations = 0;
    int releases = 0;
  } state;
  struct Allocator {
    State* state;
    Workspace allocate(std::uint64_t bytes, StreamRef stream) {
      ++state->allocations;
      state->requested = bytes;
      state->allocated_on = stream;
      return {state->storage, bytes};
    }
    void deallocate(Workspace workspace, StreamRef stream) noexcept {
      ++state->releases;
      state->released = workspace;
      state->released_on = stream;
    }
  };
  const StreamRef stream(reinterpret_cast<CUstream_st*>(&state));
  SECTION("nonempty scratch is released during unwinding") {
    REQUIRE_THROWS_AS(([&] {
                        katal::util::ScopedWorkspace scratch(Allocator{&state},
                                                             32, stream);
                        REQUIRE(scratch.get().ptr == state.storage);
                        REQUIRE(scratch.get().size == 32);
                        REQUIRE(state.releases == 0);
                        throw std::runtime_error("execution failed");
                      }()),
                      std::runtime_error);
    REQUIRE(state.allocations == 1);
    REQUIRE(state.releases == 1);
    REQUIRE(state.requested == 32);
    REQUIRE(state.released.ptr == state.storage);
    REQUIRE(state.released.size == 32);
    REQUIRE(state.allocated_on.get() == stream.get());
    REQUIRE(state.released_on.get() == stream.get());
  }
  SECTION("empty scratch needs no allocator calls") {
    {
      katal::util::ScopedWorkspace scratch(Allocator{&state}, 0, stream);
      REQUIRE(scratch.get().ptr == nullptr);
      REQUIRE(scratch.get().size == 0);
    }
    REQUIRE(state.allocations == 0);
    REQUIRE(state.releases == 0);
  }
}
