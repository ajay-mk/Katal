#ifndef CUEIN_EQUATION_HPP_
#define CUEIN_EQUATION_HPP_

#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <cuein/error.hpp>
#include <cuein/util/index.hpp>

namespace cuein {

/// The kernel an equation lowers to.
enum class Lowering { Contract, Permute, Reduce, Copy };

/// How each mode of an equation participates. The sets are named from the
/// contraction's point of view; only the ones the chosen lowering needs are
/// populated.
struct Classification {
  Lowering kind = Lowering::Contract;
  util::Index batch;       ///< in A, B and the result
  util::Index contracted;  ///< in A and B, summed away
  util::Index free_a;      ///< in A and the result only
  util::Index free_b;      ///< in B and the result only
  util::Index reduced;  ///< unary only: present in A, dropped from the result
};

/// A parsed einsum equation: the operand indices and the result index.
class Equation {
 public:
  Equation() = default;

  static Equation parse(std::string_view text);
  static Equation parse(std::string_view text, std::error_code& ec) noexcept;

  std::size_t n_operands() const noexcept { return operands_.size(); }
  std::size_t rank(std::size_t operand) const noexcept {
    return operands_[operand].rank();
  }
  const util::Index& operand(std::size_t i) const noexcept {
    return operands_[i];
  }
  const util::Index& result() const noexcept { return result_; }
  std::string_view text() const noexcept { return text_; }

  Classification classify() const;
  Classification classify(std::error_code& ec) const noexcept;

 private:
  util::Index make_index(std::string_view piece) const;
  Classification classify_unary() const;
  Classification classify_binary() const;

  /// Renders an index for a diagnostic. Multi-character labels come back
  /// space separated, which is how they were written.
  static std::string quoted(const util::Index& i) {
    return "\"" + std::string(i) + "\"";
  }

  static bool has_repeated_label(const util::Index& i) {
    for (std::size_t x = 0; x < i.rank(); ++x)
      for (std::size_t y = x + 1; y < i.rank(); ++y)
        if (i[x] == i[y]) return true;
    return false;
  }

  std::string text_;
  std::vector<util::Index> operands_;
  util::Index result_;
};

inline util::Index Equation::make_index(std::string_view piece) const {
  try {
    return util::Index(piece);
  } catch (const std::length_error& e) {
    throw Error(Errc::RankMismatch, text_, e.what());
  }
}

inline Equation Equation::parse(std::string_view text) {
  Equation eq;
  eq.text_ = std::string(text);

  std::string_view lhs = text, rhs;
  bool explicit_result = false;
  if (const auto arrow = text.find("->"); arrow != std::string_view::npos) {
    if (text.find("->", arrow + 2) != std::string_view::npos)
      throw Error(Errc::ParseError, text, "more than one '->'");
    lhs = text.substr(0, arrow);
    rhs = text.substr(arrow + 2);
    explicit_result = true;
  }

  for (std::size_t b = 0;;) {
    const auto comma = lhs.find(',', b);
    const auto piece = comma == std::string_view::npos
                           ? lhs.substr(b)
                           : lhs.substr(b, comma - b);
    eq.operands_.push_back(eq.make_index(piece));
    if (eq.operands_.back().rank() == 0)
      throw Error(Errc::ParseError, text, "an operand is empty");
    if (comma == std::string_view::npos) break;
    b = comma + 1;
  }

  // A structural character inside a label means the equation was misspelled;
  // tokenizing it would silently produce a mode named '-' or ','.
  const auto reject_structural = [&](const util::Index& i) {
    for (const util::Label& l : i)
      if (l.view().find_first_of(",->") != std::string_view::npos)
        throw Error(Errc::ParseError, text,
                    "label \"" + std::string(l.view()) +
                        "\" contains one of ',', '-' or '>'");
  };
  for (const util::Index& o : eq.operands_) reject_structural(o);

  if (explicit_result) {
    eq.result_ = eq.make_index(rhs);
    reject_structural(eq.result_);
  } else {
    // Every shared label is contracted away.
    for (const util::Index& o : eq.operands_) eq.result_ = eq.result_ ^ o;
  }
  return eq;
}

inline Equation Equation::parse(std::string_view text,
                                std::error_code& ec) noexcept {
  // ponytail: the throwing path is the implementation and this wraps it. That
  // serves -fno-exceptions callers only once these layers sit behind the
  // compiled library boundary, which is where the guarantee belongs anyway.
  try {
    ec.clear();
    return parse(text);
  } catch (const Error& e) {
    ec = make_error_code(e.code());
  } catch (...) {
    ec = make_error_code(Errc::ParseError);
  }
  return Equation{};
}

inline Classification Equation::classify_binary() const {
  const util::Index &a = operands_[0], &b = operands_[1], &d = result_;

  if (const util::Index dangling = (a ^ b) - d; dangling.rank() > 0)
    throw Error(Errc::UnsupportedPattern, text_,
                "label(s) " + quoted(dangling) +
                    " appear in one input only and not in the result; a free "
                    "reduction cannot be fused into a contraction");

  if (const util::Index broadcast = d - (a | b); broadcast.rank() > 0)
    throw Error(Errc::UnsupportedPattern, text_,
                "label(s) " + quoted(broadcast) +
                    " appear in the result but in no input; broadcasting is "
                    "not expressible");

  Classification c;
  c.kind = Lowering::Contract;
  c.batch = a & b & d;
  c.contracted = (a & b) - d;
  c.free_a = (a - b) & d;
  c.free_b = (b - a) & d;
  return c;
}

inline Classification Equation::classify_unary() const {
  const util::Index &a = operands_[0], &d = result_;

  if (const util::Index broadcast = d - a; broadcast.rank() > 0)
    throw Error(Errc::UnsupportedPattern, text_,
                "label(s) " + quoted(broadcast) +
                    " appear in the result but in no input; broadcasting is "
                    "not expressible");

  Classification c;
  c.reduced = a - d;
  if (c.reduced.rank() > 0)
    c.kind = Lowering::Reduce;
  else
    c.kind = (a == d) ? Lowering::Copy : Lowering::Permute;
  return c;
}

inline Classification Equation::classify() const {
  if (operands_.empty())
    throw Error(Errc::ParseError, text_, "the equation has no operands");

  for (const util::Index& o : operands_)
    if (has_repeated_label(o))
      throw Error(Errc::UnsupportedPattern, text_,
                  "operand " + quoted(o) +
                      " repeats a label; diagonal extraction is not supported");

  if (has_repeated_label(result_))
    throw Error(Errc::UnsupportedPattern, text_,
                "the result " + quoted(result_) +
                    " repeats a label; writing a diagonal is not expressible");

  // With at most two operands and no repeats within one, no label can occur
  // more than twice, so that rejection is the arity check below.
  if (operands_.size() == 1) return classify_unary();
  if (operands_.size() == 2) return classify_binary();
  throw Error(Errc::UnsupportedPattern, text_,
              "more than two operands requires a contraction path");
}

inline Classification Equation::classify(std::error_code& ec) const noexcept {
  try {
    ec.clear();
    return classify();
  } catch (const Error& e) {
    ec = make_error_code(e.code());
  } catch (...) {
    ec = make_error_code(Errc::UnsupportedPattern);
  }
  return Classification{};
}

}  // namespace cuein

#endif  // CUEIN_EQUATION_HPP_
