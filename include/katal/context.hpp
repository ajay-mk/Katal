#ifndef KATAL_CONTEXT_HPP_
#define KATAL_CONTEXT_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <katal/assert.hpp>
#include <katal/options.hpp>
#include <katal/scalar.hpp>
#include <katal/tensor_ref.hpp>

/// Declared exactly as the CUDA runtime declares it, so a cudaStream_t
/// converts without this header pulling in <cuda_runtime.h>.
struct CUstream_st;

namespace katal {

/// A CUDA stream Katal does not own, wrapped so that this header need not
/// include <cuda_runtime.h>. Constructs implicitly from a cudaStream_t; the
/// default is the CUDA default stream.
class StreamRef {
 public:
  StreamRef() = default;
  StreamRef(CUstream_st* stream) : stream_(stream) {}  // NOLINT: implicit
  CUstream_st* get() const noexcept { return stream_; }

 private:
  CUstream_st* stream_ = nullptr;
};

/// Process-wide settings, read once when a Context is constructed. Passing
/// this to `initialize` rather than mutating a global means there is exactly
/// one moment configuration takes effect.
struct Config {
  /// Entries in cuTENSOR's own plan cache, which stores kernel selections so a
  /// repeated `make_plan` skips the search. cuTENSOR's own default is 64; a
  /// negative value leaves it, and 0 turns the cache off.
  int32_t plan_cache_capacity = 256;

  /// How cached_plan and einsum plan through this context, unless a call
  /// overrides it. Setting a policy here rather than at each call site is what
  /// keeps the one-shot `einsum` usable while still being tunable.
  PlanOptions plan_options;

  /// Plans Katal keeps keyed on the equation and the shapes. This covers
  /// what cuTENSOR's cache cannot: its keys are the descriptors themselves, so
  /// they must be rebuilt before it can be consulted. Only the one-shot
  /// `einsum` path benefits; a hoisted Plan already skips all of it. 0 disables
  /// it. When full, the least recently used plan is evicted. The default
  /// leaves room for workloads that repeat many distinct contractions, some
  /// needing over 1,500 plans per iteration: a cache smaller than that set
  /// misses on every iteration, not just the first.
  std::size_t plan_cache_entries = 4096;

  /// When set, the device's default CUDA memory pool keeps up to this many
  /// freed bytes instead of returning them to the driver at every
  /// synchronization. CUDA's default is 0, so a loop that synchronizes each
  /// iteration remaps the einsum scratch and pooled Tensor memory every time;
  /// `UINT64_MAX` keeps it all. Applied when the Context is constructed.
  ///
  /// The pool belongs to the device, not to Katal: this changes it for every
  /// `cudaMallocAsync` in the process, and memory the pool holds is not
  /// available to `cudaMalloc`. Unset leaves the pool as it is. Constructing
  /// the Context throws `Errc::CudaError` if the device has no default pool.
  std::optional<std::uint64_t> async_pool_release_threshold;
};

/// Owns a backend, cuTENSOR by default, and the plan cache. Move-only;
/// `execute` on a plan built from it is const and stream-safe. A context serves
/// the one CUDA device that was current when it was built: planning or
/// executing with another device current throws `Errc::CudaError`. A moved-from
/// context may only be destroyed or assigned to.
class Context {
 public:
  explicit Context(Config config = {});
  Context(Context&&) noexcept;
  Context& operator=(Context&&) noexcept;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  ~Context();

  /// The CUDA device this context's backend was created on. A plan built here
  /// is tied to that device.
  int device() const;

  /// Internal. Public only so the compiled frontend can reach the backend
  /// and plan cache, and so tests can build a context around their own
  /// backend, without putting any library header in this one.
  class Impl;
  const Impl& impl() const {
    KATAL_ASSERT(impl_ != nullptr, "use of a moved-from Context");
    return *impl_;
  }

 private:
  /// For Impl, which builds a context around a backend it is handed.
  explicit Context(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

/// Sets up the process-wide context eagerly. Optional: the first call that
/// needs one does it lazily with default configuration. Calling it explicitly
/// puts the slow one-time cuTENSOR initialisation somewhere you chose.
/// Throws `Errc::InvalidState` if the context already exists, since its Config
/// could no longer take effect; call `finalize` first to change it.
void initialize(Config config = {});

/// Releases the process-wide context while the CUDA context is still alive.
/// Without this the handle is leaked on purpose, because destroying it from a
/// static destructor races CUDA's own teardown.
/// All borrowed references, Plans and GPU work must be finished first.
/// Throws `Errc::InvalidState` while a Runtime owns the default context.
void finalize();

bool initialized() noexcept;

/// Exclusive scope owner of the default context. Throws `Errc::InvalidState`
/// if a context already exists. Declare before its Plans and GPU resources, and
/// finish submitted work before leaving the scope.
class Runtime {
 public:
  explicit Runtime(Config config = {});
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
};

/// The process-wide context, built on first use if `initialize` was not
/// called. Safe to call from several threads; the reference stays valid until
/// `finalize`.
const Context& default_context();

}  // namespace katal

#endif  // KATAL_CONTEXT_HPP_
