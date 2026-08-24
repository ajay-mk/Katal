#include <string>
#include <system_error>

#include <catch2/catch_test_macros.hpp>

#include <deviceein/equation.hpp>

using deviceein::Classification;
using deviceein::Equation;
using deviceein::Errc;
using deviceein::Lowering;

namespace {

// Every rejection below must name the labels it is complaining about; a
// diagnostic that only says "invalid mode" is the thing this replaces.
bool mentions(const deviceein::Error& e, std::string_view labels) {
  return std::string(e.what()).find(labels) != std::string::npos;
}

Errc code_of(std::string_view text) {
  std::error_code ec;
  const Equation eq = Equation::parse(text, ec);
  if (ec) return static_cast<Errc>(ec.value());
  eq.classify(ec);
  return static_cast<Errc>(ec.value());
}

}  // namespace

TEST_CASE("equation parsing", "[equation]") {
  SECTION("binary contraction") {
    const Equation eq = Equation::parse("ijab,abcd->ijcd");
    REQUIRE(eq.n_operands() == 2);
    REQUIRE(eq.rank(0) == 4);
    REQUIRE(eq.rank(1) == 4);
    REQUIRE(eq.result() == deviceein::util::Index("ijcd"));
    REQUIRE(eq.text() == "ijab,abcd->ijcd");
  }

  SECTION("unary") {
    const Equation eq = Equation::parse("ijkl->ij");
    REQUIRE(eq.n_operands() == 1);
    REQUIRE(eq.result() == deviceein::util::Index("ij"));
  }

  SECTION("a rank-zero result is a full reduction, not a parse error") {
    const Equation eq = Equation::parse("ij->");
    REQUIRE(eq.result().rank() == 0);
  }

  SECTION("omitting -> contracts every shared label") {
    REQUIRE(Equation::parse("ij,jk").result() == deviceein::util::Index("ik"));
    REQUIRE(Equation::parse("ij,kl").result() ==
            deviceein::util::Index("ijkl"));
  }

  SECTION("whitespace spelling is the same equation") {
    const Equation spaced = Equation::parse("ij, jk -> ik");
    REQUIRE(spaced.n_operands() == 2);
    REQUIRE(spaced.operand(1) == deviceein::util::Index("jk"));
    REQUIRE(spaced.result() == deviceein::util::Index("ik"));
  }
}

TEST_CASE("equation parse rejections", "[equation]") {
  REQUIRE_THROWS_AS(Equation::parse(""), deviceein::Error);
  REQUIRE_THROWS_AS(Equation::parse("ij,,jk->ik"), deviceein::Error);
  REQUIRE_THROWS_AS(Equation::parse("->ij"), deviceein::Error);
  REQUIRE_THROWS_AS(Equation::parse("ij->jk->ik"), deviceein::Error);
  REQUIRE_THROWS_AS(Equation::parse("i-j,jk->ik"), deviceein::Error);

  REQUIRE(code_of("") == Errc::ParseError);
  REQUIRE(code_of("ij,,jk->ik") == Errc::ParseError);
  REQUIRE(code_of("ij->jk->ik") == Errc::ParseError);
  REQUIRE(code_of("i-j,jk->ik") == Errc::ParseError);

  SECTION("rank beyond the configured maximum") {
    const std::string wide(deviceein::max_rank + 1, 'a');
    REQUIRE(code_of(wide + "->") == Errc::RankMismatch);
  }
}

TEST_CASE("classification picks the kernel", "[equation]") {
  SECTION("binary contraction") {
    const Classification c = Equation::parse("ij,jk->ik").classify();
    REQUIRE(c.kind == Lowering::Contract);
    REQUIRE(c.batch.rank() == 0);
    REQUIRE(c.contracted == deviceein::util::Index("j"));
    REQUIRE(c.free_a == deviceein::util::Index("i"));
    REQUIRE(c.free_b == deviceein::util::Index("k"));
  }

  SECTION("batched contraction") {
    const Classification c = Equation::parse("bij,bjk->bik").classify();
    REQUIRE(c.kind == Lowering::Contract);
    REQUIRE(c.batch == deviceein::util::Index("b"));
    REQUIRE(c.contracted == deviceein::util::Index("j"));
  }

  SECTION("outer product is still a contraction") {
    const Classification c = Equation::parse("ij,kl->ijkl").classify();
    REQUIRE(c.kind == Lowering::Contract);
    REQUIRE(c.contracted.rank() == 0);
    REQUIRE(c.free_a == deviceein::util::Index("ij"));
    REQUIRE(c.free_b == deviceein::util::Index("kl"));
  }

  SECTION("permutation") {
    const Classification c = Equation::parse("ijkl->ikjl").classify();
    REQUIRE(c.kind == Lowering::Permute);
    REQUIRE(c.reduced.rank() == 0);
  }

  SECTION("reduction") {
    const Classification c = Equation::parse("ijkl->ij").classify();
    REQUIRE(c.kind == Lowering::Reduce);
    REQUIRE(c.reduced == deviceein::util::Index("kl"));
  }

  SECTION("full reduction") {
    REQUIRE(Equation::parse("ij->").classify().kind == Lowering::Reduce);
  }

  SECTION("identity is a copy, not a permutation") {
    REQUIRE(Equation::parse("ijkl->ijkl").classify().kind == Lowering::Copy);
  }
}

TEST_CASE("classification rejections name the offending labels", "[equation]") {
  SECTION("a free reduction alongside a contraction") {
    const Equation eq = Equation::parse("ijk,kl->il");
    REQUIRE_THROWS_AS(eq.classify(), deviceein::Error);
    try {
      eq.classify();
    } catch (const deviceein::Error& e) {
      REQUIRE(e.code() == Errc::UnsupportedPattern);
      REQUIRE(mentions(e, "\"j\""));
      REQUIRE(mentions(e, "ijk,kl->il"));
    }
  }

  SECTION("broadcasting a label that appears in no input") {
    const Equation eq = Equation::parse("ij,jk->ikl");
    try {
      eq.classify();
      FAIL("expected a rejection");
    } catch (const deviceein::Error& e) {
      REQUIRE(e.code() == Errc::UnsupportedPattern);
      REQUIRE(mentions(e, "\"l\""));
    }
  }

  SECTION("diagonal extraction") {
    REQUIRE(code_of("ii->i") == Errc::UnsupportedPattern);
    REQUIRE(code_of("iij,jk->ik") == Errc::UnsupportedPattern);
  }

  SECTION("writing a diagonal") {
    REQUIRE(code_of("ij,jk->ii") == Errc::UnsupportedPattern);
  }

  SECTION("more than two operands") {
    REQUIRE(code_of("ij,jk,kl->il") == Errc::UnsupportedPattern);
  }
}

TEST_CASE("the error_code overloads agree with the throwing ones",
          "[equation]") {
  std::error_code ec;
  const Equation bad = Equation::parse("ij,,jk->ik", ec);
  REQUIRE(ec == deviceein::make_error_code(Errc::ParseError));
  REQUIRE(bad.n_operands() == 0);
  REQUIRE(ec.message() == "malformed equation");
  REQUIRE(std::string(ec.category().name()) == "deviceein");

  ec.clear();
  const Equation good = Equation::parse("ij,jk->ik", ec);
  REQUIRE_FALSE(ec);
  good.classify(ec);
  REQUIRE_FALSE(ec);

  ec.clear();
  Equation::parse("ijk,kl->il").classify(ec);
  REQUIRE(ec == deviceein::make_error_code(Errc::UnsupportedPattern));
}
