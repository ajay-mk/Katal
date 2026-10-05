#ifndef KATAL_TEST_THROWN_HPP_
#define KATAL_TEST_THROWN_HPP_

#include <optional>

#include <katal/error.hpp>

namespace katal::test {

/// The Errc a call throws, or nullopt when it throws something else or
/// nothing. Pins both the exception type and its code in one comparison.
template <class F>
std::optional<Errc> thrown_code(F&& f) {
  try {
    f();
  } catch (const Error& e) {
    return e.code();
  } catch (...) {
  }
  return std::nullopt;
}

}  // namespace katal::test

#endif  // KATAL_TEST_THROWN_HPP_
