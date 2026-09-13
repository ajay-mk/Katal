#ifndef KATAL_SRC_CONTEXT_IMPL_HPP_
#define KATAL_SRC_CONTEXT_IMPL_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "cutensor_status.hpp"

#include <katal/context.hpp>
#include <katal/plan.hpp>

namespace katal {

/// Shared between the context and plan translation units; never visible to a
/// caller, which is what keeps <cutensor.h> out of the public headers.
class Context::Impl {
 public:
  explicit Impl(Config config) : config_(std::move(config)) {
    KATAL_CHECK(cutensorCreate(&handle_), "");
    std::unique_ptr<std::remove_pointer_t<cutensorHandle_t>,
                    decltype(&cutensorDestroy)>
        guard(handle_, &cutensorDestroy);
    if (config_.plan_cache_capacity > 0)
      KATAL_CHECK(
          cutensorHandleResizePlanCache(
              handle_, static_cast<uint32_t>(config_.plan_cache_capacity)),
          "");
    KATAL_CHECK_CUDA(cudaGetDevice(&device_), "");

    // A missing or stale cache file just means the first run pays for kernel
    // selection, so a failure to read is not worth reporting.
    if (!config_.plan_cache_file.empty()) {
      uint32_t lines = 0;
      cutensorHandleReadPlanCacheFromFile(
          handle_, config_.plan_cache_file.c_str(), &lines);
    }
    guard.release();
  }

  ~Impl() {
    // Destructors are noexcept and swallow errors; there is nothing useful to
    // do with a failure here.
    if (handle_ == nullptr) return;
    plans_.clear();
    if (!config_.plan_cache_file.empty())
      cutensorHandleWritePlanCacheToFile(handle_,
                                         config_.plan_cache_file.c_str());
    cutensorDestroy(handle_);
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  cutensorHandle_t handle() const noexcept { return handle_; }
  int device() const noexcept { return device_; }
  const Config& config() const noexcept { return config_; }

  /// Looks a plan up by key, building it with `make` on a miss. Locked, so a
  /// plan may be shared across threads; execute is const and stream-safe.
  ///
  /// When full the whole cache is dropped rather than evicting one entry: a
  /// handful of shapes is the normal case, and outstanding shared_ptrs keep
  /// any plan in use alive.
  template <class Make>
  std::shared_ptr<const Plan> cached(const std::string& key, Make make) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (config_.plan_cache_entries == 0)
      return std::make_shared<const Plan>(make());
    if (const auto it = plans_.find(key); it != plans_.end()) return it->second;
    auto plan = std::make_shared<const Plan>(make());
    if (plans_.size() >= config_.plan_cache_entries) plans_.clear();
    plans_.emplace(key, plan);
    return plan;
  }

 private:
  cutensorHandle_t handle_ = nullptr;
  int device_ = 0;
  Config config_;
  mutable std::mutex mutex_;
  mutable std::unordered_map<std::string, std::shared_ptr<const Plan>> plans_;
};

}  // namespace katal

#endif  // KATAL_SRC_CONTEXT_IMPL_HPP_
