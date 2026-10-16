#ifndef KATAL_SRC_CONTEXT_IMPL_HPP_
#define KATAL_SRC_CONTEXT_IMPL_HPP_

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "backend.hpp"
#include "log.hpp"

#include <katal/context.hpp>
#include <katal/error.hpp>
#include <katal/plan.hpp>

namespace katal {

namespace detail {

/// Sets the release threshold of `device`'s default memory pool; see
/// Config::async_pool_release_threshold. Defined with the CUDA allocator.
void set_async_pool_release_threshold(int device, std::uint64_t bytes);

}  // namespace detail

/// Shared between the context and plan translation units; never visible to a
/// caller. It owns the backend and the plan cache. cuTENSOR is the default
/// backend, named only through make_cutensor_backend, which keeps <cutensor.h>
/// out of the public headers and out of the frontend.
class Context::Impl {
 public:
  explicit Impl(Config config)
      : config_(std::move(config)),
        backend_(backend::make_cutensor_backend(config_)) {}

  Impl(Config config, std::unique_ptr<backend::Backend> backend)
      : config_(std::move(config)), backend_(non_null(std::move(backend))) {}

  /// A context that plans and runs through `backend` instead of cuTENSOR,
  /// which is how the host tests watch what the frontend lowers.
  static Context with_backend(std::unique_ptr<backend::Backend> backend,
                              Config config = {}) {
    return Context(
        std::make_unique<Impl>(std::move(config), std::move(backend)));
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  const backend::Backend& backend() const noexcept { return *backend_; }
  int device() const noexcept { return backend_->device(); }
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
  std::shared_ptr<const Plan> cached(std::string key, Make make) const {
    // A hit skips planning, which is where a plan checks its device, so check
    // here too: a plan from this cache is no use on another device.
    check_device();
    if (config_.plan_cache_entries == 0) {
      KATAL_LOG("cache.bypass device=" << device() << " key="
                                       << std::hash<std::string>{}(key));
      return std::make_shared<const Plan>(make());
    }
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (auto hit = touch(key)) {
        KATAL_LOG("cache.hit device="
                  << device() << " key=" << std::hash<std::string>{}(key)
                  << " equation=" << util::log_quote(hit->equation()));
        return hit;
      }
    }
    KATAL_LOG("cache.miss device=" << device()
                                   << " key=" << std::hash<std::string>{}(key));
    auto plan = std::make_shared<const Plan>(make());

    const std::lock_guard<std::mutex> lock(mutex_);
    if (auto raced = touch(key)) return raced;
    plans_.emplace_front(std::move(key), plan);
    try {
      index_.emplace(plans_.front().first, plans_.begin());
    } catch (...) {
      plans_.pop_front();
      throw;
    }
    if (plans_.size() > config_.plan_cache_entries) {
      KATAL_LOG("cache.evict device="
                << device() << " key="
                << std::hash<std::string>{}(plans_.back().first) << " equation="
                << util::log_quote(plans_.back().second->equation()));
      index_.erase(plans_.back().first);
      plans_.pop_back();
    }
    KATAL_LOG("cache.insert device="
              << device()
              << " key=" << std::hash<std::string>{}(plans_.front().first)
              << " equation=" << util::log_quote(plan->equation())
              << " entries=" << plans_.size());
    return plan;
  }

 private:
  void check_device() const { backend_->check_device("", "context"); }

  using Entry = std::pair<std::string, std::shared_ptr<const Plan>>;

  static std::unique_ptr<backend::Backend> non_null(
      std::unique_ptr<backend::Backend> backend) {
    if (backend == nullptr)
      throw Error(Errc::InvalidState, "", "a context needs a backend");
    return backend;
  }

  /// The cached plan for `key`, moved to the front as the most recently used,
  /// or null. Requires the lock.
  std::shared_ptr<const Plan> touch(std::string_view key) const {
    const auto it = index_.find(key);
    if (it == index_.end()) return nullptr;
    plans_.splice(plans_.begin(), plans_, it->second);
    return it->second->second;
  }

  // Declared in initialization order: the backend reads the config. Cached
  // plans borrow the backend, so the cache is declared after it and destroyed
  // first.
  Config config_;
  std::unique_ptr<backend::Backend> backend_;
  mutable std::mutex mutex_;
  /// Most recently used first. Splicing keeps the iterators in `index_` valid.
  mutable std::list<Entry> plans_;
  /// Keys view the strings owned by `plans_`, whose nodes never move, so an
  /// entry is erased here before its node is popped.
  mutable std::unordered_map<std::string_view, std::list<Entry>::iterator>
      index_;
};

}  // namespace katal

#endif  // KATAL_SRC_CONTEXT_IMPL_HPP_
