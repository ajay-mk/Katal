#include <string>

#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <katal/assert.hpp>

namespace {

/// Counts its calls, so a test can see whether a condition was evaluated.
bool counted(int& calls, bool result) {
  ++calls;
  return result;
}

}  // namespace

TEST_CASE("a passing KATAL_ASSERT does nothing", "[assert]") {
  int calls = 0;
  KATAL_ASSERT(counted(calls, true));
  KATAL_ASSERT(counted(calls, true), "with a message");
  if constexpr (katal::assert_behavior == katal::AssertBehavior::Ignore)
    CHECK(calls == 0);
  else
    CHECK(calls == 2);
}

TEST_CASE("a failing KATAL_ASSERT follows the configured behavior",
          "[assert]") {
  int calls = 0;
  if constexpr (katal::assert_behavior == katal::AssertBehavior::Throw) {
    try {
      KATAL_ASSERT(counted(calls, false), "the message");
      FAIL("KATAL_ASSERT did not throw");
    } catch (const katal::Error& e) {
      CHECK(e.code() == katal::Errc::InvalidState);
      CHECK(e.equation().empty());
      using Catch::Matchers::ContainsSubstring;
      CHECK_THAT(e.what(),
                 ContainsSubstring("KATAL_ASSERT(counted(calls, false)) "
                                   "failed: the message"));
      CHECK_THAT(e.what(), ContainsSubstring("test_assert.cpp:"));
    }
    CHECK(calls == 1);
  } else if constexpr (katal::assert_behavior ==
                       katal::AssertBehavior::Ignore) {
    KATAL_ASSERT(counted(calls, false), "the message");
    CHECK(calls == 0);
  } else {
    // A failing assertion aborts in this build, which a test cannot survive.
    // Not SKIP: an older catch_discover_tests reads a skipped case as failed.
    SUCCEED("a failing assertion aborts in this build");
  }
}

TEST_CASE("KATAL_ENFORCE throws its code whatever the assert behavior",
          "[assert]") {
  int details = 0;
  const auto detail = [&details] {
    ++details;
    return std::string("the detail");
  };

  KATAL_ENFORCE(true, katal::Errc::WorkspaceTooSmall, "ij->ji", detail());
  CHECK(details == 0);

  try {
    KATAL_ENFORCE(1 + 1 == 3, katal::Errc::WorkspaceTooSmall, "ij->ji",
                  detail());
    FAIL("KATAL_ENFORCE did not throw");
  } catch (const katal::Error& e) {
    CHECK(e.code() == katal::Errc::WorkspaceTooSmall);
    CHECK(e.equation() == "ij->ji");
    CHECK(std::string(e.what()) == "katal: in \"ij->ji\": the detail");
  }
  CHECK(details == 1);
}
