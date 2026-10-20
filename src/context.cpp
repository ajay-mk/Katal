#include <atomic>
#include <mutex>
#include <utility>

#include "context_impl.hpp"

#include <katal/context.hpp>
#include <katal/error.hpp>

namespace katal {

Context::Context(Config config) : impl_(std::make_unique<Impl>(config)) {
  if (const auto threshold = impl_->config().async_pool_release_threshold)
    detail::set_async_pool_release_threshold(impl_->device(), *threshold);
}
Context::Context(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;
Context::~Context() = default;

int Context::device() const { return impl().device(); }

namespace {

std::mutex& process_mutex() {
  static std::mutex mutex;
  return mutex;
}

// A raw owning pointer on purpose: a smart pointer here would destroy the
// handle from a static destructor, racing CUDA's own teardown. Without a
// finalize() call the handle leaks and the operating system reclaims it.
// Atomic so default_context(), which every einsum call evaluates, can read it
// without the mutex; every write still happens under process_mutex().
std::atomic<Context*> g_context = nullptr;
Config g_config;
bool g_runtime_active = false;

}  // namespace

void initialize(Config config) {
  const std::lock_guard<std::mutex> lock(process_mutex());
  if (g_context.load(std::memory_order_relaxed) != nullptr)
    throw Error(Errc::InvalidState, "",
                "the default context already exists, so this Config cannot "
                "take effect; call finalize() first");
  g_config = config;
  g_context.store(new Context(std::move(config)), std::memory_order_release);
}

void finalize() {
  const std::lock_guard<std::mutex> lock(process_mutex());
  if (g_runtime_active)
    throw Error(Errc::InvalidState, "", "Runtime owns the default context");
  delete g_context.exchange(nullptr, std::memory_order_acq_rel);
}

Runtime::Runtime(Config config) {
  const std::lock_guard<std::mutex> lock(process_mutex());
  if (g_context.load(std::memory_order_relaxed) != nullptr)
    throw Error(Errc::InvalidState, "", "the default context already exists");
  g_config = config;
  g_context.store(new Context(std::move(config)), std::memory_order_release);
  g_runtime_active = true;
}

Runtime::~Runtime() {
  const std::lock_guard<std::mutex> lock(process_mutex());
  delete g_context.exchange(nullptr, std::memory_order_acq_rel);
  g_runtime_active = false;
}

bool initialized() noexcept {
  return g_context.load(std::memory_order_acquire) != nullptr;
}

const Context& default_context() {
  // The common case once the context exists. Racing finalize() is already
  // outside the contract: the reference was never valid past it.
  if (Context* context = g_context.load(std::memory_order_acquire))
    return *context;
  const std::lock_guard<std::mutex> lock(process_mutex());
  Context* context = g_context.load(std::memory_order_relaxed);
  if (context == nullptr) {
    context = new Context(g_config);
    g_context.store(context, std::memory_order_release);
  }
  return *context;
}

}  // namespace katal
