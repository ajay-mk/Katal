#include <mutex>
#include <utility>

#include "context_impl.hpp"

#include <katal/context.hpp>
#include <katal/error.hpp>

namespace katal {

Context::Context(Config config) : impl_(std::make_unique<Impl>(config)) {}
Context::Context(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;
Context::~Context() = default;

int Context::device() const noexcept { return impl().device(); }

namespace {

std::mutex& process_mutex() {
  static std::mutex mutex;
  return mutex;
}

// A raw owning pointer on purpose: a smart pointer here would destroy the
// handle from a static destructor, racing CUDA's own teardown. Without a
// finalize() call the handle leaks and the operating system reclaims it.
Context* g_context = nullptr;
Config g_config;
bool g_runtime_active = false;

}  // namespace

void initialize(Config config) {
  const std::lock_guard<std::mutex> lock(process_mutex());
  if (g_context != nullptr)
    throw Error(Errc::InvalidState, "",
                "the default context already exists, so this Config cannot "
                "take effect; call finalize() first");
  g_config = config;
  g_context = new Context(std::move(config));
}

void finalize() {
  const std::lock_guard<std::mutex> lock(process_mutex());
  if (g_runtime_active)
    throw Error(Errc::InvalidState, "", "Runtime owns the default context");
  delete g_context;
  g_context = nullptr;
}

Runtime::Runtime(Config config) {
  const std::lock_guard<std::mutex> lock(process_mutex());
  if (g_context != nullptr)
    throw Error(Errc::InvalidState, "", "the default context already exists");
  g_config = config;
  g_context = new Context(std::move(config));
  g_runtime_active = true;
}

Runtime::~Runtime() {
  const std::lock_guard<std::mutex> lock(process_mutex());
  delete g_context;
  g_context = nullptr;
  g_runtime_active = false;
}

bool initialized() noexcept {
  const std::lock_guard<std::mutex> lock(process_mutex());
  return g_context != nullptr;
}

const Context& default_context() {
  const std::lock_guard<std::mutex> lock(process_mutex());
  if (g_context == nullptr) g_context = new Context(g_config);
  return *g_context;
}

}  // namespace katal
