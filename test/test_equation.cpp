#include <cstddef>
#include <optional>
#include <string>

#include "enum_names.hpp"
#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>

#include <katal/equation.hpp>

using katal::Classification;
using katal::Equation;
using katal::Errc;
using katal::Lowering;
using katal::test::thrown_code;

namespace {

// Every rejection below must name the labels it is complaining about; a
// diagnostic that only says "invalid mode" is the thing this replaces.
bool mentions(const katal::Error& e, std::string_view labels) {
  return std::string(e.what()).find(labels) != std::string::npos;
}

std::optional<Errc> code_of(std::string_view text) {
  return thrown_code([&] { Equation::parse(text).classify(); });
}

// max_rank labels, none shared with another block. Bytes from 'A' upward
// avoid the separators and whitespace, and above 127 are still labels.
std::string distinct_labels(std::size_t block) {
  if (katal::max_rank > (256 - 'A') / 2)
    SKIP("two disjoint maximum-rank operands need more byte labels");
  std::string labels;
  for (std::size_t i = 0; i < katal::max_rank; ++i)
    labels += static_cast<char>('A' + block * katal::max_rank + i);
  return labels;
}

}  // namespace

TEST_CASE("equation parsing", "[equation]") {
  SECTION("binary contraction") {
    const Equation eq = Equation::parse("ijab,abcd->ijcd");
    REQUIRE(eq.n_operands() == 2);
    REQUIRE(eq.rank(0) == 4);
    REQUIRE(eq.rank(1) == 4);
    REQUIRE(eq.result() == katal::util::Index("ijcd"));
    REQUIRE(eq.text() == "ijab,abcd->ijcd");
  }

  SECTION("unary") {
    const Equation eq = Equation::parse("ijkl->ij");
    REQUIRE(eq.n_operands() == 1);
    REQUIRE(eq.result() == katal::util::Index("ij"));
  }

  SECTION("a rank-zero result is a full reduction, not a parse error") {
    const Equation eq = Equation::parse("ij->");
    REQUIRE(eq.result().rank() == 0);
  }

  SECTION("omitting -> keeps the labels that appear once, sorted as NumPy") {
    REQUIRE(Equation::parse("ij,jk").result() == katal::util::Index("ik"));
    REQUIRE(Equation::parse("ij,kl").result() == katal::util::Index("ijkl"));
    REQUIRE(Equation::parse("kj,ji").result() == katal::util::Index("ik"));
    REQUIRE(Equation::parse("lk,ij").result() == katal::util::Index("ijkl"));
    // By byte value, so upper case sorts first.
    REQUIRE(Equation::parse("ba,AB").result() == katal::util::Index("ABab"));
  }

  SECTION("an implicit unary result is sorted, so it can transpose") {
    const Equation eq = Equation::parse("ba");
    REQUIRE(eq.result() == katal::util::Index("ab"));
    REQUIRE(eq.classify().kind == Lowering::Permute);
    REQUIRE(Equation::parse("ab").classify().kind == Lowering::Copy);
  }

  SECTION("whitespace spelling is the same equation") {
    const Equation spaced = Equation::parse("ij, jk -> ik");
    REQUIRE(spaced.n_operands() == 2);
    REQUIRE(spaced.operand(1) == katal::util::Index("jk"));
    REQUIRE(spaced.result() == katal::util::Index("ik"));
  }

  SECTION("every ASCII whitespace character is skipped") {
    const Equation eq = Equation::parse("ij,\n\tjk\r\n->\v\fik\n");
    REQUIRE(eq.operand(0) == katal::util::Index("ij"));
    REQUIRE(eq.operand(1) == katal::util::Index("jk"));
    REQUIRE(eq.result() == katal::util::Index("ik"));
    REQUIRE(code_of("ij,\njk->ik") == std::nullopt);
  }
}

TEST_CASE("equation parse rejections", "[equation]") {
  REQUIRE_THROWS_AS(Equation::parse(""), katal::Error);
  REQUIRE_THROWS_AS(Equation::parse("ij,,jk->ik"), katal::Error);
  REQUIRE_THROWS_AS(Equation::parse("->ij"), katal::Error);
  REQUIRE_THROWS_AS(Equation::parse("ij->jk->ik"), katal::Error);
  REQUIRE_THROWS_AS(Equation::parse("i-j,jk->ik"), katal::Error);

  REQUIRE(code_of("") == Errc::ParseError);
  REQUIRE(code_of("ij,,jk->ik") == Errc::ParseError);
  REQUIRE(code_of("ij->jk->ik") == Errc::ParseError);
  REQUIRE(code_of("i-j,jk->ik") == Errc::ParseError);

  SECTION("rank beyond the configured maximum") {
    const std::string wide(katal::max_rank + 1, 'a');
    REQUIRE(code_of(wide + "->") == Errc::RankMismatch);
  }

  SECTION("an implicit result beyond the maximum rank is a RankMismatch") {
    // Two operands at the maximum rank with no label in common.
    const std::string a = distinct_labels(0), b = distinct_labels(1);
    const std::string text = a + "," + b;
    try {
      Equation::parse(text);
      FAIL("expected a rejection");
    } catch (const katal::Error& e) {
      REQUIRE(e.code() == Errc::RankMismatch);
      REQUIRE(e.equation() == text);
    }
    REQUIRE(code_of(text) == Errc::RankMismatch);
  }
}

TEST_CASE("classification picks the kernel", "[equation]") {
  SECTION("the tensor rank limit does not limit the union of labels") {
    if (katal::max_rank < 9) SKIP("needs rank-nine tensors");
    const auto c = Equation::parse("abcdefghi,efghijklm->abcdjklm").classify();
    REQUIRE(c.kind == Lowering::Contract);
    REQUIRE(c.contracted == katal::util::Index("efghi"));
    REQUIRE(c.free_a == katal::util::Index("abcd"));
    REQUIRE(c.free_b == katal::util::Index("jklm"));
  }
  SECTION("binary contraction") {
    const Classification c = Equation::parse("ij,jk->ik").classify();
    REQUIRE(c.kind == Lowering::Contract);
    REQUIRE(c.batch.rank() == 0);
    REQUIRE(c.contracted == katal::util::Index("j"));
    REQUIRE(c.free_a == katal::util::Index("i"));
    REQUIRE(c.free_b == katal::util::Index("k"));
  }

  SECTION("batched contraction") {
    const Classification c = Equation::parse("bij,bjk->bik").classify();
    REQUIRE(c.kind == Lowering::Contract);
    REQUIRE(c.batch == katal::util::Index("b"));
    REQUIRE(c.contracted == katal::util::Index("j"));
  }

  SECTION("outer product is still a contraction") {
    const Classification c = Equation::parse("ij,kl->ijkl").classify();
    REQUIRE(c.kind == Lowering::Contract);
    REQUIRE(c.contracted.rank() == 0);
    REQUIRE(c.free_a == katal::util::Index("ij"));
    REQUIRE(c.free_b == katal::util::Index("kl"));
  }

  SECTION("permutation") {
    const Classification c = Equation::parse("ijkl->ikjl").classify();
    REQUIRE(c.kind == Lowering::Permute);
    REQUIRE(c.reduced.rank() == 0);
  }

  SECTION("reduction") {
    const Classification c = Equation::parse("ijkl->ij").classify();
    REQUIRE(c.kind == Lowering::Reduce);
    REQUIRE(c.reduced == katal::util::Index("kl"));
  }

  SECTION("full reduction") {
    REQUIRE(Equation::parse("ij->").classify().kind == Lowering::Reduce);
  }

  SECTION("identity is a copy, not a permutation") {
    REQUIRE(Equation::parse("ijkl->ijkl").classify().kind == Lowering::Copy);
  }
}

TEST_CASE("classification rejections name the offending labels", "[equation]") {
  SECTION("broadcasting a label that appears in no input") {
    const Equation eq = Equation::parse("ij,jk->ikl");
    try {
      eq.classify();
      FAIL("expected a rejection");
    } catch (const katal::Error& e) {
      REQUIRE(e.code() == Errc::UnsupportedPattern);
      REQUIRE(mentions(e, "\"l\""));
    }
  }

  SECTION("writing a diagonal") {
    REQUIRE(code_of("ij,jk->ii") == Errc::UnsupportedPattern);
  }

  SECTION("more than two operands") {
    REQUIRE(code_of("ij,jk,kl->il") == Errc::UnsupportedPattern);
  }
}

TEST_CASE("elementwise classification", "[equation]") {
  SECTION("matching labels in any order") {
    const Classification c =
        Equation::parse("jiab,ijab->ijab").classify_elementwise();
    REQUIRE(c.kind == Lowering::Elementwise);
    REQUIRE(c.batch == katal::util::Index("jiab"));
    REQUIRE(c.broadcast.rank() == 0);
  }

  SECTION("an input lacking a result label is broadcast along it") {
    const Classification c =
        Equation::parse("ijab,ab->ijab").classify_elementwise();
    REQUIRE(c.batch == katal::util::Index("ab"));
    REQUIRE(c.broadcast == katal::util::Index("ij"));
  }

  SECTION("a result label in neither input is broadcast from both") {
    const Classification c = Equation::parse("i,j->ijk").classify_elementwise();
    REQUIRE(c.batch.rank() == 0);
    REQUIRE(c.broadcast == katal::util::Index("ijk"));
  }

  SECTION("an equation einsum would contract is not elementwise") {
    try {
      Equation::parse("ij,jk->ik").classify_elementwise();
      FAIL("expected a rejection");
    } catch (const katal::Error& e) {
      REQUIRE(e.code() == Errc::UnsupportedPattern);
      REQUIRE(mentions(e, "label(s) \"j\" appear"));
    }
  }

  SECTION("summed labels from both inputs can exceed one index") {
    const std::string text =
        distinct_labels(0) + "," + distinct_labels(1) + "->";
    REQUIRE(thrown_code([&] {
              Equation::parse(text).classify_elementwise();
            }) == Errc::UnsupportedPattern);
  }

  SECTION("one or three inputs") {
    REQUIRE(thrown_code([] {
              Equation::parse("ij->ji").classify_elementwise();
            }) == Errc::UnsupportedPattern);
    REQUIRE(thrown_code([] {
              Equation::parse("ij,ij,ij->ij").classify_elementwise();
            }) == Errc::UnsupportedPattern);
  }

  SECTION("a repeated input label reads the diagonal") {
    const Classification c =
        Equation::parse("ii,ij->ij").classify_elementwise();
    REQUIRE(c.diagonal == katal::util::Index("i"));
    REQUIRE(c.batch == katal::util::Index("i"));
  }
}

TEST_CASE("permutation classification", "[equation]") {
  SECTION("a reordering") {
    const Classification c = Equation::parse("ijk->kij").classify_permutation();
    REQUIRE(c.kind == Lowering::Permute);
    REQUIRE(c.broadcast.rank() == 0);
  }

  SECTION("the same order is a copy") {
    REQUIRE(Equation::parse("ij->ij").classify_permutation().kind ==
            Lowering::Copy);
  }

  SECTION("a result label the input lacks is broadcast") {
    const Classification c = Equation::parse("j->ij").classify_permutation();
    REQUIRE(c.kind == Lowering::Permute);
    REQUIRE(c.broadcast == katal::util::Index("i"));
  }

  SECTION("a dropped label is a reduction, not a permutation") {
    try {
      Equation::parse("ijk->ik").classify_permutation();
      FAIL("expected a rejection");
    } catch (const katal::Error& e) {
      REQUIRE(e.code() == Errc::UnsupportedPattern);
      REQUIRE(mentions(e, "\"j\""));
    }
  }

  SECTION("two inputs") {
    REQUIRE(thrown_code([] {
              Equation::parse("ij,jk->ik").classify_permutation();
            }) == Errc::UnsupportedPattern);
  }
}

TEST_CASE("diagonals and free reductions", "[equation]") {
  SECTION("a trace") {
    const Classification c = Equation::parse("ii->").classify();
    REQUIRE(c.kind == Lowering::Reduce);
    REQUIRE(c.reduced == katal::util::Index("i"));
    REQUIRE(c.diagonal == katal::util::Index("i"));
  }

  SECTION("the implicit result of a repeated label is a trace too") {
    const Equation eq = Equation::parse("ii");
    REQUIRE(eq.result().rank() == 0);
    REQUIRE(eq.classify().kind == Lowering::Reduce);
  }

  SECTION("a diagonal") {
    const Classification c = Equation::parse("ii->i").classify();
    REQUIRE(c.kind == Lowering::Copy);
    REQUIRE(c.diagonal == katal::util::Index("i"));
  }

  SECTION("a diagonal inside a contraction") {
    const Classification c = Equation::parse("iij,jk->ik").classify();
    REQUIRE(c.kind == Lowering::Contract);
    REQUIRE(c.diagonal == katal::util::Index("i"));
    REQUIRE(c.free_a == katal::util::Index("i"));
    REQUIRE(c.contracted == katal::util::Index("j"));
  }

  SECTION("a label in one input alone is summed out of it") {
    const Classification c = Equation::parse("ijk,kl->il").classify();
    REQUIRE(c.kind == Lowering::Contract);
    REQUIRE(c.reduced_a == katal::util::Index("j"));
    REQUIRE(c.reduced_b.rank() == 0);
    REQUIRE(c.contracted == katal::util::Index("k"));
  }

  SECTION("free reductions on both inputs beyond the maximum rank") {
    // Together the reduced labels do not fit in one index; kept per input,
    // each does.
    const std::string a = distinct_labels(0), b = distinct_labels(1);
    const Classification c = Equation::parse(a + "," + b + "->A").classify();
    REQUIRE(c.reduced_a == katal::util::Index(a.substr(1)));
    REQUIRE(c.reduced_b == katal::util::Index(b));
  }

  SECTION("a repeated result label is still rejected") {
    REQUIRE(code_of("ij,jk->ii") == Errc::UnsupportedPattern);
  }
}
