#ifndef DEVICEEIN_CONTEXT_HPP_
#define DEVICEEIN_CONTEXT_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

#include <deviceein/options.hpp>
#include <deviceein/scalar.hpp>
#include <deviceein/tensor_ref.hpp>

/// Declared exactly as the CUDA runtime declares it, so a cudaStream_t
/// converts without this header pulling in <cuda_runtime.h>.
struct CUstream_st;

namespace deviceein {

/// A CUDA stream DeviceEin does not own, wrapped so that this header need not
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
  /// repeated `make_plan` skips the search. cuTENSOR's own default is 64.
  int32_t plan_cache_capacity = 256;

  /// Where to persist that cache between runs. Read when a Context is built
  /// and written when it is destroyed, so the first run of a program pays for
  /// kernel selection and later ones do not. Empty disables it. A missing file
  /// is not an error; an unreadable one is ignored.
  std::filesystem::path plan_cache_file;

  /// How every plan built through this context is planned, unless a call
  /// overrides it. Setting a policy here rather than at each call site is what
  /// keeps the one-shot `einsum` usable while still being tunable.
  PlanOptions plan_options;

  /// Plans DeviceEin keeps keyed on the equation and the shapes. This covers
  /// what cuTENSOR's cache cannot: its keys are the descriptors themselves, so
  /// they must be rebuilt before it can be consulted. Only the one-shot
  /// `einsum` path benefits; a hoisted Plan already skips all of it. 0 disables
  /// it.
  std::size_t plan_cache_entries = 128;
};

/// Owns a cuTENSOR handle. Move-only; `execute` on a plan built from it is
/// const and stream-safe.
class Context {
 public:
  explicit Context(Config config = {});
  Context(Context&&) noexcept;
  Context& operator=(Context&&) noexcept;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  ~Context();

  /// The CUDA device this context's handle was created on. A plan built here
  /// is tied to that device.
  int device() const noexcept;

  /// Internal. Public only so the compiled glue can reach the handle without
  /// putting <cutensor.h> in this header.
  class Impl;
  const Impl& impl() const noexcept { return *impl_; }

 private:
  std::unique_ptr<Impl> impl_;
};

/// Sets up the process-wide context eagerly. Optional: the first call that
/// needs one does it lazily with default configuration. Calling it explicitly
/// puts the slow one-time cuTENSOR initialisation somewhere you chose.
void initialize(Config config = {});

/// Releases the process-wide context while the CUDA context is still alive.
/// Without this the handle is leaked on purpose, because destroying it from a
/// static destructor races CUDA's own teardown.
void finalize();

bool initialized() noexcept;

/// Scope guard over the pair above. Equivalent, and cannot be forgotten.
class Runtime {
 public:
  explicit Runtime(Config config = {}) { initialize(config); }
  ~Runtime() { finalize(); }
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
};

/// The process-wide context, built on first use if `initialize` was not
/// called. Safe to call from several threads; the reference stays valid until
/// `finalize`.
const Context& default_context();

}  // namespace deviceein

#endif  // DEVICEEIN_CONTEXT_HPP_
