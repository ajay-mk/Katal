#ifndef KATAL_EQUATION_HPP_
#define KATAL_EQUATION_HPP_

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <katal/error.hpp>
#include <katal/util/index.hpp>

namespace katal {

/// The kernel an equation lowers to.
enum class Lowering {
  Contract,
  Permute,
  Reduce,
  Copy,
  Elementwise,
  /// Three inputs contracted in one kernel, `e = alpha * A.B.C + beta * d`.
  ContractTrinary,
};

/// How each mode of an equation participates. The sets are named from the
/// binary contraction's point of view; only the ones the chosen lowering needs
/// are populated. A three-operand contraction fills `batch`, `contracted`,
/// `reduced_a`, `reduced_b`, `reduced_c` and `diagonal`.
struct Classification {
  Lowering kind = Lowering::Contract;
  util::Index batch;  ///< in every input and the result
  /// In two or more inputs (A and B for a binary contraction), summed away.
  util::Index contracted;
  util::Index free_a;   ///< in A and the result only
  util::Index free_b;   ///< in B and the result only
  util::Index reduced;  ///< unary only: present in A, dropped from the result
  /// Binary only: in A alone, summed out of A before the contraction.
  util::Index reduced_a;
  /// Binary only: in B alone, summed out of B before the contraction.
  util::Index reduced_b;
  /// Three operands only: in C alone, summed out of C before the contraction.
  /// `reduced_a` and `reduced_b` hold A's and B's such labels there too.
  util::Index reduced_c;
  /// Repeated within an input, which is read along that diagonal: `"ii->"`
  /// is a trace, `"ii->i"` a diagonal.
  util::Index diagonal;
  /// Elementwise only: in the result but missing from an input, which is
  /// broadcast along it.
  util::Index broadcast;
};

/// A parsed einsum equation: the operand indices and the result index.
class Equation {
 public:
  Equation() = default;

  /// Throws `Error` with `Errc::ParseError` or `Errc::RankMismatch`.
  static Equation parse(std::string_view text);

  std::size_t n_operands() const noexcept { return operands_.size(); }

  /// Number of labels on one operand, which must match that tensor's extents.
  std::size_t rank(std::size_t operand) const noexcept {
    return operands_[operand].rank();
  }

  /// Labels of one operand, in the order written.
  const util::Index& operand(std::size_t i) const noexcept {
    return operands_[i];
  }

  /// Labels of the result: as written after `->`. When the equation omits it,
  /// the labels that appear exactly once across the operands, sorted by byte
  /// value as NumPy does, so `"kj,ji"` gives `"ik"` and `"ba"` a transpose.
  const util::Index& result() const noexcept { return result_; }

  /// The equation exactly as given, which is what error messages quote.
  std::string_view text() const noexcept { return text_; }

  /// Works out how each mode participates and which kernel the equation
  /// lowers to. Throws `Errc::UnsupportedPattern` for anything Katal
  /// declines to lower, naming the labels responsible.
  Classification classify() const;

  /// Checks the equation as an elementwise operation on two inputs, which
  /// pairs elements by label rather than summing over shared ones. A label an
  /// input lacks is broadcast; every input label must appear in the result.
  /// Throws `Errc::UnsupportedPattern` otherwise, naming the labels.
  Classification classify_elementwise() const;

  /// Checks the equation as a permutation of one input, which may also
  /// broadcast it along result labels it lacks but sums over nothing. The
  /// kind is `Permute`, or `Copy` when the result repeats the input.
  /// Throws `Errc::UnsupportedPattern` otherwise, naming the labels.
  Classification classify_permutation() const;

 private:
  util::Index make_index(std::string_view piece) const;
  util::Index implicit_result() const;
  Classification classify_unary() const;
  Classification classify_binary() const;
  Classification classify_trinary() const;
  void check_result() const;

  /// Renders an index for a diagnostic, in quotes.
  static std::string quoted(const util::Index& i) {
    return "\"" + std::string(i) + "\"";
  }

  /// Each label once, in first-seen order: an operand with its diagonal
  /// labels merged, which is how every classification sees it.
  static util::Index unique(const util::Index& i) {
    util::Index u;
    for (const util::Label l : i)
      if (!u.contains(l)) u.push_back(l);
    return u;
  }

  /// Labels that occur more than once in `i`, each listed once.
  static util::Index repeated(const util::Index& i) {
    util::Index r;
    for (std::size_t x = 0; x < i.rank(); ++x)
      for (std::size_t y = x + 1; y < i.rank(); ++y)
        if (i[x] == i[y] && !r.contains(i[x])) r.push_back(i[x]);
    return r;
  }

  Classification with_diagonal(Classification c) const {
    for (const util::Index& o : operands_)
      c.diagonal = c.diagonal | repeated(o);
    return c;
  }

  std::string text_;
  std::vector<util::Index> operands_;
  util::Index result_;
};

inline util::Index Equation::make_index(std::string_view piece) const {
  try {
    return util::Index(piece);
  } catch (const Error&) {
    throw Error(Errc::RankMismatch, text_,
                "an operand or the result has more than " +
                    std::to_string(max_rank) + " labels");
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
    for (const util::Label l : i)
      if (l == ',' || l == '-' || l == '>')
        throw Error(Errc::ParseError, text,
                    "label \"" + std::string(1, l) +
                        "\" contains one of ',', '-' or '>'");
  };
  for (const util::Index& o : eq.operands_) reject_structural(o);

  if (explicit_result) {
    eq.result_ = eq.make_index(rhs);
    reject_structural(eq.result_);
  } else {
    eq.result_ = eq.implicit_result();
  }
  return eq;
}

inline util::Index Equation::implicit_result() const {
  // Indexed by byte, so walking it in order is the sort.
  std::array<std::size_t, 256> count{};
  for (const util::Index& o : operands_)
    for (const util::Label l : o) ++count[static_cast<unsigned char>(l)];

  util::Index r;
  for (std::size_t b = 0; b < count.size(); ++b) {
    if (count[b] != 1) continue;
    if (r.rank() == max_rank)
      throw Error(Errc::RankMismatch, text_,
                  "the implicit result has more than " +
                      std::to_string(max_rank) +
                      " labels; write the result after '->'");
    r.push_back(static_cast<util::Label>(b));
  }
  return r;
}

inline Classification Equation::classify_binary() const {
  const util::Index a = unique(operands_[0]), b = unique(operands_[1]);
  const util::Index& d = result_;

  if (const util::Index broadcast = (d - a) - b; broadcast.rank() > 0)
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
  // Kept per input: the two together can carry more labels than one index
  // holds.
  c.reduced_a = (a - b) - d;
  c.reduced_b = (b - a) - d;
  return with_diagonal(c);
}

inline Classification Equation::classify_trinary() const {
  const util::Index a = unique(operands_[0]), b = unique(operands_[1]),
                    c = unique(operands_[2]);
  const util::Index& d = result_;

  if (const util::Index broadcast = ((d - a) - b) - c; broadcast.rank() > 0)
    throw Error(Errc::UnsupportedPattern, text_,
                "label(s) " + quoted(broadcast) +
                    " appear in the result but in no input; broadcasting is "
                    "not expressible");

  Classification r;
  r.kind = Lowering::ContractTrinary;
  r.batch = a & b & c & d;
  // Each pair's shared labels, minus those the result keeps. Unlike a binary
  // contraction's, these need not fit in one input, so they can outgrow an
  // index.
  try {
    r.contracted = ((a & b) - d) | ((a & c) - d) | ((b & c) - d);
  } catch (const Error&) {
    throw Error(Errc::UnsupportedPattern, text_,
                "more than " + std::to_string(max_rank) +
                    " labels are summed across the three inputs");
  }
  // Kept per input, as in classify_binary.
  r.reduced_a = ((a - b) - c) - d;
  r.reduced_b = ((b - a) - c) - d;
  r.reduced_c = ((c - a) - b) - d;
  return with_diagonal(r);
}

inline Classification Equation::classify_unary() const {
  const util::Index a = unique(operands_[0]);
  const util::Index& d = result_;

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
  return with_diagonal(c);
}

inline void Equation::check_result() const {
  if (operands_.empty())
    throw Error(Errc::ParseError, text_, "the equation has no operands");

  if (repeated(result_).rank() > 0)
    throw Error(Errc::UnsupportedPattern, text_,
                "the result " + quoted(result_) +
                    " repeats a label; writing a diagonal is not expressible");
}

inline Classification Equation::classify() const {
  check_result();

  if (operands_.size() == 1) return classify_unary();
  if (operands_.size() == 2) return classify_binary();
  if (operands_.size() == 3) return classify_trinary();
  throw Error(Errc::UnsupportedPattern, text_,
              "more than three operands requires a contraction path");
}

inline Classification Equation::classify_elementwise() const {
  check_result();
  if (operands_.size() != 2)
    throw Error(Errc::UnsupportedPattern, text_,
                "an elementwise operation takes two inputs, not " +
                    std::to_string(operands_.size()));

  const util::Index a = unique(operands_[0]), b = unique(operands_[1]);
  const util::Index& d = result_;
  // Checked and rendered per input, as in classify_binary: the two together
  // can carry more labels than one index holds.
  const util::Index summed_a = a - d, summed_b = b - d;
  if (summed_a.rank() > 0 || summed_b.rank() > 0)
    throw Error(Errc::UnsupportedPattern, text_,
                "label(s) \"" + std::string(summed_a) +
                    std::string(summed_b - summed_a) + "\"" +
                    " appear in an input but not in the result; an "
                    "elementwise operation sums over nothing, so use einsum");

  Classification c;
  c.kind = Lowering::Elementwise;
  c.batch = a & b;
  c.broadcast = d - (a & b);
  return with_diagonal(c);
}

inline Classification Equation::classify_permutation() const {
  check_result();
  if (operands_.size() != 1)
    throw Error(Errc::UnsupportedPattern, text_,
                "a permutation takes one input, not " +
                    std::to_string(operands_.size()));

  const util::Index a = unique(operands_[0]);
  const util::Index& d = result_;
  if (const util::Index summed = a - d; summed.rank() > 0)
    throw Error(Errc::UnsupportedPattern, text_,
                "label(s) " + quoted(summed) +
                    " appear in the input but not in the result; a "
                    "permutation sums over nothing, so use einsum or reduce");

  Classification c;
  c.kind = (a == d) ? Lowering::Copy : Lowering::Permute;
  c.broadcast = d - a;
  return with_diagonal(c);
}

}  // namespace katal

#endif  // KATAL_EQUATION_HPP_
