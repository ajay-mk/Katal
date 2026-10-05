#ifndef KATAL_SRC_CONTEXT_IMPL_HPP_
#define KATAL_SRC_CONTEXT_IMPL_HPP_

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "cuda_status.hpp"
#include "cutensor_status.hpp"
#include "log.hpp"

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
    handle_ = guard.release();
  }

  ~Impl() {
    // Destructors are noexcept and swallow errors; there is nothing useful to
    // do with a failure here.
    if (handle_ == nullptr) return;
    // cuTENSOR objects must be destroyed on the device they were made on,
    // and the caller may have another device current by now.
    util::ScopedDevice on_device(device_);
    index_.clear();
    plans_.clear();
    cutensorDestroy(handle_);
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  cutensorHandle_t handle() const noexcept { return handle_; }
  int device() const noexcept { return device_; }
  const Config& config() const noexcept { return config_; }

  /// Looks a plan up by key, building it with `make` on a miss. A plan may be
  /// shared across threads; execute is const and stream-safe.
  ///
  /// Planning runs outside the lock, since with JIT it can take seconds and
  /// every other thread's lookup would wait on it. Two threads that miss on
  /// the same key both plan; the first to insert wins and both return its
  /// plan. When full, the least recently used plan is evicted; outstanding
  /// shared_ptrs keep any plan in use alive.
  template <class Make>
  std::shared_ptr<const Plan> cached(const std::string& key, Make make) const {
    // A hit skips planning, which is where a plan checks its device, so check
    // here too: a plan from this cache is no use on another device.
    check_device();
    if (config_.plan_cache_entries == 0) {
      KATAL_LOG("cache.bypass device=" << device_ << " key="
                                       << std::hash<std::string>{}(key));
      return std::make_shared<const Plan>(make());
    }
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (auto hit = touch(key)) {
        KATAL_LOG("cache.hit device="
                  << device_ << " key=" << std::hash<std::string>{}(key)
                  << " equation=" << util::log_quote(hit->equation()));
        return hit;
      }
    }
    KATAL_LOG("cache.miss device=" << device_
                                   << " key=" << std::hash<std::string>{}(key));
    auto plan = std::make_shared<const Plan>(make());

    const std::lock_guard<std::mutex> lock(mutex_);
    if (auto raced = touch(key)) return raced;
    if (plans_.size() >= config_.plan_cache_entries) {
      KATAL_LOG(
          "cache.evict device="
          << device_ << " key=" << std::hash<std::string>{}(plans_.back().first)
          << " equation=" << util::log_quote(plans_.back().second->equation()));
      index_.erase(plans_.back().first);
      plans_.pop_back();
    }
    plans_.emplace_front(key, plan);
    index_.emplace(key, plans_.begin());
    KATAL_LOG("cache.insert device="
              << device_ << " key=" << std::hash<std::string>{}(key)
              << " equation=" << util::log_quote(plan->equation())
              << " entries=" << plans_.size());
    return plan;
  }

 private:
  void check_device() const {
    int current = 0;
    KATAL_CHECK_CUDA(cudaGetDevice(&current), "");
    if (current != device_)
      throw Error(Errc::CudaError, "",
                  "this context belongs to device " + std::to_string(device_) +
                      " but device " + std::to_string(current) + " is current");
  }

  using Entry = std::pair<std::string, std::shared_ptr<const Plan>>;

  /// The cached plan for `key`, moved to the front as the most recently used,
  /// or null. Requires the lock.
  std::shared_ptr<const Plan> touch(const std::string& key) const {
    const auto it = index_.find(key);
    if (it == index_.end()) return nullptr;
    plans_.splice(plans_.begin(), plans_, it->second);
    return it->second->second;
  }

  cutensorHandle_t handle_ = nullptr;
  int device_ = 0;
  Config config_;
  mutable std::mutex mutex_;
  /// Most recently used first. Splicing keeps the iterators in `index_` valid.
  mutable std::list<Entry> plans_;
  mutable std::unordered_map<std::string, std::list<Entry>::iterator> index_;
};

}  // namespace katal

#endif  // KATAL_SRC_CONTEXT_IMPL_HPP_
