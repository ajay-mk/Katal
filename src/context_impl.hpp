#ifndef CUEIN_SRC_CONTEXT_IMPL_HPP_
#define CUEIN_SRC_CONTEXT_IMPL_HPP_

#include "cutensor_status.hpp"

#include <cuein/context.hpp>

namespace cuein {

/// Shared between the context and plan translation units; never visible to a
/// caller, which is what keeps <cutensor.h> out of the public headers.
class Context::Impl {
 public:
  explicit Impl(Config config) {
    CUEIN_CHECK(cutensorCreate(&handle_), "");
    if (config.plan_cache_capacity > 0)
      CUEIN_CHECK(
          cutensorHandleResizePlanCache(
              handle_, static_cast<uint32_t>(config.plan_cache_capacity)),
          "");
    CUEIN_CHECK_CUDA(cudaGetDevice(&device_), "");
  }

  ~Impl() {
    // Destructors are noexcept and swallow errors; there is nothing useful to
    // do with a failure here.
    if (handle_ != nullptr) cutensorDestroy(handle_);
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  cutensorHandle_t handle() const noexcept { return handle_; }
  int device() const noexcept { return device_; }

 private:
  cutensorHandle_t handle_ = nullptr;
  int device_ = 0;
};

}  // namespace cuein

#endif  // CUEIN_SRC_CONTEXT_IMPL_HPP_
