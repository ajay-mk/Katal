#ifndef KATAL_SRC_CONTEXT_IMPL_HPP_
#define KATAL_SRC_CONTEXT_IMPL_HPP_

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "backend.hpp"
#include "log.hpp"

#include <katal/context.hpp>
#include <katal/plan.hpp>

namespace katal {

/// Shared between the context and plan translation units; never visible to a
/// caller. It owns the backend and the plan cache, and names no library, which
/// keeps <cutensor.h> out of the public headers and out of the front end.
class Context::Impl {
 public:
  explicit Impl(Config config)
      : config_(std::move(config)),
        backend_(backend::make_cutensor_backend(config_)),
        device_(backend_->device()) {}

  // Cached plans borrow the backend, so they go first.
  ~Impl() {
    index_.clear();
    plans_.clear();
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  const backend::Backend& backend() const noexcept { return *backend_; }
  int device() const noexcept { return device_; }
  const Config& config() const noexcept { return config_; }

  /// Looks a plan up by key, building it with `make` on a miss. A plan may be
  /// shared across threads; execute is const and stream-safe.
  ///
  /// Planning runs outside the lock, since with JIT it can take seconds and
  /// every other thread's lookup would wait on it. Two threads that miss on
  /// the same key may both plan; insertion reuses a matching plan if it is
  /// still cached. When full, the least recently used plan is evicted;
  /// outstanding shared_ptrs keep any plan in use alive.
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
    plans_.emplace_front(key, plan);
    try {
      index_.emplace(key, plans_.begin());
    } catch (...) {
      plans_.pop_front();
      throw;
    }
    if (plans_.size() > config_.plan_cache_entries) {
      KATAL_LOG(
          "cache.evict device="
          << device_ << " key=" << std::hash<std::string>{}(plans_.back().first)
          << " equation=" << util::log_quote(plans_.back().second->equation()));
      index_.erase(plans_.back().first);
      plans_.pop_back();
    }
    KATAL_LOG("cache.insert device="
              << device_ << " key=" << std::hash<std::string>{}(key)
              << " equation=" << util::log_quote(plan->equation())
              << " entries=" << plans_.size());
    return plan;
  }

 private:
  void check_device() const { backend_->check_device("", "context"); }

  using Entry = std::pair<std::string, std::shared_ptr<const Plan>>;

  /// The cached plan for `key`, moved to the front as the most recently used,
  /// or null. Requires the lock.
  std::shared_ptr<const Plan> touch(const std::string& key) const {
    const auto it = index_.find(key);
    if (it == index_.end()) return nullptr;
    plans_.splice(plans_.begin(), plans_, it->second);
    return it->second->second;
  }

  // Declared in initialization order: the backend reads the config.
  Config config_;
  std::unique_ptr<backend::Backend> backend_;
  int device_ = 0;
  mutable std::mutex mutex_;
  /// Most recently used first. Splicing keeps the iterators in `index_` valid.
  mutable std::list<Entry> plans_;
  mutable std::unordered_map<std::string, std::list<Entry>::iterator> index_;
};

}  // namespace katal

#endif  // KATAL_SRC_CONTEXT_IMPL_HPP_
