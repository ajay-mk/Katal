#ifndef KATAL_ASSERT_HPP_
#define KATAL_ASSERT_HPP_

#include <cstdio>
#include <cstdlib>
#include <source_location>
#include <string>

#include <katal/config.hpp>
#include <katal/error.hpp>

namespace katal {

/// What a failed `KATAL_ASSERT` does, fixed when Katal is configured by
/// `-DKATAL_ASSERT_BEHAVIOR`.
enum class AssertBehavior {
  /// Throw `katal::Error` with `Errc::InvalidState`.
  Throw = KATAL_ASSERT_THROW,
  /// Print the failed check to stderr and call `std::abort`.
  Abort = KATAL_ASSERT_ABORT,
  /// Compile the check out. The condition is not evaluated.
  Ignore = KATAL_ASSERT_IGNORE,
};

/// The behavior this Katal was configured with.
inline constexpr AssertBehavior assert_behavior =
    static_cast<AssertBehavior>(KATAL_ASSERT_BEHAVIOR);

namespace detail {

/// Reports a failed `KATAL_ASSERT`. `check` names the condition and carries
/// the message; `where` is the assertion's own location.
[[noreturn]] inline void assert_failed(
    const char* check,
    std::source_location where = std::source_location::current()) {
  if constexpr (assert_behavior == AssertBehavior::Throw) {
    std::string detail = check;
    detail += " (";
    detail += where.file_name();
    detail += ':';
    detail += std::to_string(where.line());
    detail += ')';
    throw Error(Errc::InvalidState, {}, detail);
  } else {
    // No allocation here: aborting must not depend on the heap.
    std::fprintf(stderr, "katal: %s (%s:%u)\n", check, where.file_name(),
                 static_cast<unsigned>(where.line()));
    std::abort();
  }
}

}  // namespace detail
}  // namespace katal

/// The text a failed KATAL_ASSERT reports. The optional message must be a
/// string literal.
#define KATAL_ASSERT_MESSAGE_(EXPR, ...) \
  "KATAL_ASSERT(" #EXPR ") failed" __VA_OPT__(": " __VA_ARGS__)

/// Checks something Katal itself guarantees, or a misuse the API documents as
/// undefined, such as using a moved-from Context. A failure is a bug, either
/// in Katal or in how it is used, so what happens is chosen when Katal is
/// configured: see `katal::AssertBehavior`. Under `IGNORE` the condition is
/// still compiled but never evaluated, so it must not have side effects that
/// matter.
///
///     KATAL_ASSERT(impl_ != nullptr, "use of a moved-from Plan");
#if KATAL_ASSERT_BEHAVIOR == KATAL_ASSERT_IGNORE
#define KATAL_ASSERT(EXPR, ...) static_cast<void>(sizeof(!(EXPR)))
#else
#define KATAL_ASSERT(EXPR, ...)                              \
  (static_cast<bool>(EXPR) ? static_cast<void>(0)            \
                           : ::katal::detail::assert_failed( \
                                 KATAL_ASSERT_MESSAGE_(EXPR, __VA_ARGS__)))
#endif

/// Checks what a caller passed in. It is never compiled out and does not
/// follow `KATAL_ASSERT_BEHAVIOR`: a failure always throws
/// `katal::Error(CODE, EQUATION, DETAIL)`, since callers are promised that
/// code. `DETAIL` is evaluated only when the check fails, so it may build a
/// message.
///
///     KATAL_ENFORCE(workspace.size >= needed, Errc::WorkspaceTooSmall,
///                   equation_, "the workspace is too small");
#define KATAL_ENFORCE(EXPR, CODE, EQUATION, DETAIL)       \
  do {                                                    \
    if (!static_cast<bool>(EXPR)) [[unlikely]]            \
      throw ::katal::Error((CODE), (EQUATION), (DETAIL)); \
  } while (false)

#endif  // KATAL_ASSERT_HPP_
