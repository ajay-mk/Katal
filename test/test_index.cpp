#include <string>

#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>

#include <katal/util/index.hpp>

using katal::test::thrown_code;

// POSIX <strings.h> declares an `index` function at global scope, so the
// using-declarations below sit inside each test case, where they hide it.

TEST_CASE("index tokenization", "[index]") {
  using katal::util::Index;

  SECTION("labels are single characters by default") {
    REQUIRE(Index("ijab").rank() == 4);
    REQUIRE(Index("ijab").contains('i'));
    REQUIRE_FALSE(Index("ijab").contains('z'));
    REQUIRE(std::string(Index("ijab")) == "ijab");
  }

  SECTION("whitespace anywhere is ignored") {
    // This is what lets "ij, jk -> ik" mean the same as "ij,jk->ik".
    REQUIRE(Index(" jk ").rank() == 2);
    REQUIRE(Index(" jk ").contains('j'));
    REQUIRE(Index("  i j  ").rank() == 2);
    REQUIRE(Index("i j") == Index("ij"));
  }

  SECTION("empty and blank indices have rank zero") {
    REQUIRE(Index("").rank() == 0);
    REQUIRE(Index("   ").rank() == 0);
  }

  SECTION("repeated labels are representable") {
    // The equation layer rejects them; the index type does not.
    REQUIRE(Index("ii").rank() == 2);
  }
}

// The order of these results is the mode order handed to the descriptor. Every
// assertion here fails if an operator sorts.
TEST_CASE("set operators preserve the left operand's order", "[index]") {
  using katal::util::Index;

  REQUIRE((Index("kji") & Index("ijk")) == Index("kji"));
  REQUIRE((Index("ji") | Index("kj")) == Index("jik"));
  REQUIRE((Index("ijk") - Index("jk")) == Index("i"));
  REQUIRE((Index("ij") ^ Index("jk")) == Index("ik"));

  SECTION("concatenation keeps duplicates, the set operators do not") {
    REQUIRE((Index("ij") + Index("jk")) == Index("ijjk"));
    REQUIRE((Index("ij") + Index("jk")).rank() == 4);
    REQUIRE((Index("iij") & Index("ij")) == Index("ij"));
    REQUIRE((Index("iij") | Index("j")) == Index("ij"));
    REQUIRE((Index("iij") - Index("j")) == Index("i"));
  }
}

TEST_CASE("mode classification algebra", "[index]") {
  using katal::util::Index;

  SECTION("ij,jk->ik") {
    const Index a("ij"), b("jk"), d("ik");
    REQUIRE((a & b & d).rank() == 0);      // batch
    REQUIRE(((a & b) - d) == Index("j"));  // contracted
    REQUIRE(((a - b) & d) == Index("i"));  // free in A
    REQUIRE(((b - a) & d) == Index("k"));  // free in B
    REQUIRE(((a ^ b) - d).rank() == 0);    // dangling
    REQUIRE((d - (a | b)).rank() == 0);    // broadcast
  }

  SECTION("bij,bjk->bik batches b rather than contracting it") {
    const Index a("bij"), b("bjk"), d("bik");
    REQUIRE((a & b & d) == Index("b"));
    REQUIRE(((a & b) - d) == Index("j"));
  }

  SECTION("ijk,kl->il leaves j dangling") {
    const Index a("ijk"), b("kl"), d("il");
    REQUIRE(((a ^ b) - d) == Index("j"));
  }

  SECTION("ij,jk->ikl asks to broadcast l") {
    const Index a("ij"), b("jk"), d("ikl");
    REQUIRE((d - (a | b)) == Index("l"));
  }
}

TEST_CASE("permute_from_to", "[index]") {
  using katal::util::Index;
  using katal::util::Permutation;
  using katal::util::permute_from_to;

  SECTION("identity") {
    REQUIRE(permute_from_to(Index("ijk"), Index("ijk")) ==
            Permutation{0, 1, 2});
  }

  SECTION("maps each destination slot to its source slot") {
    const Index from("ijk"), to("kij");
    const Permutation p = permute_from_to(from, to);
    REQUIRE(p == Permutation{2, 0, 1});
    for (std::size_t i = 0; i < to.rank(); ++i) REQUIRE(to[i] == from[p[i]]);
  }

  SECTION("repeated labels consume distinct source slots") {
    REQUIRE(permute_from_to(Index("iij"), Index("jii")) ==
            Permutation{2, 0, 1});
  }

  SECTION("rejects indices that are not permutations of each other") {
    REQUIRE(thrown_code([] { permute_from_to(Index("ijk"), Index("ij")); }) ==
            katal::Errc::RankMismatch);
    REQUIRE(thrown_code([] { permute_from_to(Index("ijk"), Index("ijl")); }) ==
            katal::Errc::UnsupportedPattern);
  }
}

TEST_CASE("index capacity", "[index]") {
  using katal::util::Index;

  SECTION("rank beyond the configured maximum is rejected") {
    const std::string too_many(katal::max_rank + 1, 'a');
    REQUIRE(thrown_code([&] { Index{too_many}; }) == katal::Errc::RankMismatch);
  }

  SECTION("labels longer than a label can hold are rejected") {
    REQUIRE(thrown_code([] { Index{"i extremelylonglabel"}; }) ==
            katal::Errc::RankMismatch);
  }
}

// Owning no memory is the point: an index survives as a constexpr variable,
// not just as transient constant evaluation.
constexpr katal::util::Index occ_virt("ijab");
static_assert(occ_virt.rank() == 4);
static_assert(occ_virt.contains('a'));

static_assert(katal::util::Index("ijab").rank() == 4);
static_assert(katal::util::Index(" i j k ").rank() == 3);
static_assert((katal::util::Index("ij") ^ katal::util::Index("jk")) ==
              katal::util::Index("ik"));
static_assert(katal::util::permute_from_to(katal::util::Index("ijk"),
                                           katal::util::Index("kij"))[0] == 2);
