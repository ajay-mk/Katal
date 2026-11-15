#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

#include "enum_names.hpp"
#include "path.hpp"
#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>

#include <katal/equation.hpp>

using katal::Dtype;
using katal::Equation;
using katal::Errc;
using katal::detail::lower_path;
using katal::detail::Path;
using katal::detail::PathPair;
using katal::test::thrown_code;

namespace {

// Holds the extents so the spans lower_path takes stay valid.
struct Shapes {
  std::vector<std::vector<std::int64_t>> extents;
  std::vector<std::span<const std::int64_t>> spans;

  Shapes(std::initializer_list<std::vector<std::int64_t>> list)
      : extents(list) {
    for (const auto& e : extents) spans.emplace_back(e);
  }
};

Path lower(std::string_view text, const Shapes& shapes,
           std::vector<PathPair> path, Dtype type = Dtype::F64) {
  return lower_path(Equation::parse(text), shapes.spans, path, type);
}

std::vector<std::string> equations(const Path& p) {
  std::vector<std::string> out;
  for (const auto& s : p.steps) out.push_back(s.equation);
  return out;
}

}  // namespace

TEST_CASE("two operands lower to one step and need no workspace", "[path]") {
  const Path p = lower("ik,kj->ij", {{2, 3}, {3, 5}}, {{0, 1}});
  REQUIRE(equations(p) == std::vector<std::string>{"ik,kj->ij"});
  CHECK(p.steps[0].a == 0);
  CHECK(p.steps[0].b == 1);
  CHECK(p.steps[0].bytes == 0);
  CHECK(p.steps[0].multiply_adds == 2.0 * 3 * 5);
  CHECK(p.workspace == 0);
}

TEST_CASE("the path decides which intermediate is built", "[path]") {
  // A ladder-like term through an intermediate: the order picks between an
  // o^4 intermediate and a v^4 one.
  const std::int64_t o = 4, v = 10;
  const Shapes shapes{{o, o, v, v}, {o, o, v, v}, {o, o, v, v}};
  const std::string text = "klcd,ijcd,klab->ijab";

  SECTION("hole-hole intermediate first") {
    const Path p = lower(text, shapes, {{0, 1}, {0, 1}});
    REQUIRE(equations(p) ==
            std::vector<std::string>{"klcd,ijcd->klij", "klab,klij->ijab"});
    CHECK(p.steps[0].a == 0);
    CHECK(p.steps[0].b == 1);
    CHECK(p.steps[1].a == 2);
    CHECK(p.steps[1].b == 3);  // the intermediate
    CHECK(p.steps[0].bytes == static_cast<std::uint64_t>(o * o * o * o * 8));
    CHECK(p.workspace == p.steps[0].bytes);
    const double cost = static_cast<double>(o * o * o * o * v * v);
    CHECK(p.multiply_adds == 2 * cost);
  }

  SECTION("particle-particle intermediate first") {
    const Path p = lower(text, shapes, {{0, 2}, {0, 1}});
    REQUIRE(equations(p) ==
            std::vector<std::string>{"klcd,klab->cdab", "ijcd,cdab->ijab"});
    CHECK(p.steps[0].bytes == static_cast<std::uint64_t>(v * v * v * v * 8));
    CHECK(p.workspace == p.steps[0].bytes);
    const double cost = static_cast<double>(o * o * v * v * v * v);
    CHECK(p.multiply_adds == 2 * cost);
  }

  SECTION("element size follows the given type") {
    const Path p = lower(text, shapes, {{0, 1}, {0, 1}}, Dtype::C64);
    CHECK(p.steps[0].bytes == static_cast<std::uint64_t>(o * o * o * o * 16));
  }
}

TEST_CASE("intermediates keep only the labels still needed", "[path]") {
  SECTION("a triples-like term") {
    const Path p =
        lower("ijae,ekbc,kl->ijlabc", {{2, 3, 5, 7}, {7, 4, 5, 5}, {4, 3}},
              {{0, 1}, {0, 1}});
    REQUIRE(equations(p) ==
            std::vector<std::string>{"ijae,ekbc->ijakbc", "kl,ijakbc->ijlabc"});
    CHECK(p.steps[0].bytes == 2u * 3 * 5 * 4 * 5 * 5 * 8);
  }

  SECTION("a diagonal input is read along its diagonal") {
    const Path p =
        lower("iij,jk,kl->il", {{3, 3, 4}, {4, 5}, {5, 6}}, {{0, 1}, {0, 1}});
    REQUIRE(equations(p) ==
            std::vector<std::string>{"iij,jk->ik", "kl,ik->il"});
    CHECK(p.steps[0].multiply_adds == 3.0 * 4 * 5);
  }

  SECTION("a pair with no shared label is an outer product") {
    const Path p =
        lower("ij,kl,jl->ik", {{2, 3}, {4, 5}, {3, 5}}, {{0, 1}, {0, 1}});
    REQUIRE(equations(p) ==
            std::vector<std::string>{"ij,kl->ijkl", "jl,ijkl->ik"});
    CHECK(p.steps[0].multiply_adds == 2.0 * 3 * 4 * 5);
  }

  SECTION("a label only one input carries is summed in its first step") {
    const Path p =
        lower("abx,bc,cd->ad", {{2, 3, 7}, {3, 4}, {4, 5}}, {{0, 1}, {0, 1}});
    REQUIRE(equations(p) ==
            std::vector<std::string>{"abx,bc->ac", "cd,ac->ad"});
  }
}

TEST_CASE("intermediates whose lifetimes do not overlap share bytes",
          "[path]") {
  // ab,bc -> I5 (steps 0..1), cd,I5 -> I6 (1..2), de,I6 -> I7 (2..3), then
  // ef,I7 -> af. I5 is dead once I7 is written, so I7 reuses its bytes.
  const Shapes shapes{{4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 4}};
  const Path p =
      lower("ab,bc,cd,de,ef->af", shapes, {{0, 1}, {0, 3}, {0, 2}, {0, 1}});
  REQUIRE(equations(p) == std::vector<std::string>{"ab,bc->ac", "cd,ac->da",
                                                   "de,da->ea", "ef,ea->af"});
  const std::uint64_t bytes = 4 * 4 * 8;
  CHECK(p.steps[0].offset == 0);
  CHECK(p.steps[1].offset == katal::detail::path_alignment);
  CHECK(p.steps[2].offset == 0);
  CHECK(p.workspace == katal::detail::path_alignment + bytes);
  for (const auto& s : p.steps)
    CHECK(s.offset % katal::detail::path_alignment == 0);
}

TEST_CASE("two intermediates read by the same step are both kept", "[path]") {
  // (ab,bc) and (cd,de) are both live when the last step reads them.
  const Shapes shapes{{4, 4}, {4, 4}, {4, 4}, {4, 4}};
  const Path p = lower("ab,bc,cd,de->ae", shapes, {{0, 1}, {0, 1}, {0, 1}});
  REQUIRE(equations(p) ==
          std::vector<std::string>{"ab,bc->ac", "cd,de->ce", "ac,ce->ae"});
  CHECK(p.steps[0].offset == 0);
  CHECK(p.steps[1].offset == katal::detail::path_alignment);
  CHECK(p.workspace == katal::detail::path_alignment + 4 * 4 * 8);
}

TEST_CASE("bad paths and shapes are refused", "[path]") {
  const Shapes three{{2, 3}, {3, 4}, {4, 5}};
  const auto code = [](auto&& f) { return thrown_code(f); };

  CHECK(code([&] { lower("ab,bc,cd->ad", three, {{0, 1}}); }) ==
        Errc::UnsupportedPattern);
  CHECK(code([&] { lower("ab,bc,cd->ad", three, {{0, 1}, {0, 1}, {0, 1}}); }) ==
        Errc::UnsupportedPattern);
  CHECK(code([&] { lower("ab,bc,cd->ad", three, {{0, 0}, {0, 1}}); }) ==
        Errc::UnsupportedPattern);
  CHECK(code([&] { lower("ab,bc,cd->ad", three, {{0, 3}, {0, 1}}); }) ==
        Errc::UnsupportedPattern);
  // After the first step only two operands remain.
  CHECK(code([&] { lower("ab,bc,cd->ad", three, {{0, 1}, {0, 2}}); }) ==
        Errc::UnsupportedPattern);
  CHECK(code([&] { lower("ab->ba", {{2, 3}}, {}); }) ==
        Errc::UnsupportedPattern);

  CHECK(code([&] {
          lower("ab,bc,cd->ad", {{2, 3}, {3, 4}}, {{0, 1}, {0, 1}});
        }) == Errc::RankMismatch);
  CHECK(code([&] {
          lower("ab,bc,cd->ad", {{2, 3}, {3}, {4, 5}}, {{0, 1}, {0, 1}});
        }) == Errc::RankMismatch);
  CHECK(code([&] {
          lower("ab,bc,cd->ad", {{2, 3}, {9, 4}, {4, 5}}, {{0, 1}, {0, 1}});
        }) == Errc::ExtentMismatch);
  CHECK(code([&] {
          lower("ab,bc,cd->ad", {{2, 0}, {0, 4}, {4, 5}}, {{0, 1}, {0, 1}});
        }) == Errc::ExtentMismatch);
}

TEST_CASE("a step Katal cannot lower is named in the error", "[path]") {
  // The result repeats a label, which the last step cannot write.
  try {
    lower("ab,bc,cd->aa", {{2, 3}, {3, 4}, {4, 2}}, {{0, 1}, {0, 1}});
    FAIL("expected a throw");
  } catch (const katal::Error& e) {
    CHECK(e.code() == Errc::UnsupportedPattern);
    CHECK(std::string(e.what()).find("path step 1") != std::string::npos);
  }
}

TEST_CASE("an intermediate too large to address is refused", "[path]") {
  const std::int64_t huge = std::int64_t{1} << 40;
  CHECK(thrown_code([&] {
          lower("ab,cd,bd->ac", {{huge, 2}, {huge, 2}, {2, 2}},
                {{0, 1}, {0, 1}});
        }) == Errc::ExtentMismatch);
}
