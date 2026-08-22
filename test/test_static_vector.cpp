#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include <cuein/util/static_vector.hpp>

TEST_CASE("static_vector", "[static_vector]") {
  using vec = cuein::util::StaticVector<int, 4>;

  SECTION("starts empty at its fixed capacity") {
    const vec v;
    REQUIRE(v.empty());
    REQUIRE(v.size() == 0);
    REQUIRE(vec::capacity() == 4);
  }

  SECTION("push_back appends in order") {
    vec v;
    v.push_back(7);
    v.push_back(9);
    REQUIRE(v.size() == 2);
    REQUIRE(v[0] == 7);
    REQUIRE(v[1] == 9);
  }

  SECTION("compares contents, so a trailing zero is an element") {
    REQUIRE(vec{1, 2} == vec{1, 2});
    REQUIRE_FALSE(vec{1, 2} == vec{1, 2, 0});
    REQUIRE_FALSE(vec{1, 2} == vec{2, 1});
  }

  SECTION("throws instead of growing") {
    vec v{1, 2, 3, 4};
    REQUIRE_THROWS_AS(v.push_back(5), std::length_error);
  }
}

// Owning no memory is the point: unlike std::vector these survive as constexpr
// variables, not just as transient constant evaluation.
constexpr cuein::util::StaticVector<int, 4> constexpr_vector{1, 2, 3};
static_assert(constexpr_vector.size() == 3);
static_assert(constexpr_vector[2] == 3);
static_assert([] {
  cuein::util::StaticVector<int, 4> v;
  v.push_back(3);
  v.push_back(5);
  return v.size() == 2 && v[1] == 5;
}());
