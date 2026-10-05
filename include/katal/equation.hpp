#ifndef KATAL_EQUATION_HPP_
#define KATAL_EQUATION_HPP_

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <katal/error.hpp>
#include <katal/util/index.hpp>

namespace katal {

/// The kernel an equation lowers to.
enum class Lowering { Contract, Permute, Reduce, Copy, Elementwise };

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

  /// Same grammar, reporting through `ec` instead. Returns an equation with no
  /// operands on failure.
  static Equation parse(std::string_view text, std::error_code& ec) noexcept;

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

  /// Same, reporting through `ec`.
  Classification classify(std::error_code& ec) const noexcept;

  /// Checks the equation as an elementwise operation on two inputs, which
  /// pairs elements by label rather than summing over shared ones. A label an
  /// input lacks is broadcast; every input label must appear in the result.
  /// Throws `Errc::UnsupportedPattern` otherwise, naming the labels.
  Classification classify_elementwise() const;

  /// Same, reporting through `ec`.
  Classification classify_elementwise(std::error_code& ec) const noexcept;

 private:
  util::Index make_index(std::string_view piece) const;
  util::Index implicit_result() const;
  Classification classify_unary() const;
  Classification classify_binary() const;
  void check_repeats() const;

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

inline Equation Equation::parse(std::string_view text,
                                std::error_code& ec) noexcept {
  // The throwing path is the implementation and this wraps it, so no
  // exception escapes. That serves an -fno-exceptions caller only once these
  // layers sit behind the compiled library boundary.
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

  // Checked per input: the two together can carry more labels than one index
  // holds.
  const util::Index dangling_a = (a - b) - d, dangling_b = (b - a) - d;
  if (dangling_a.rank() > 0 || dangling_b.rank() > 0)
    throw Error(Errc::UnsupportedPattern, text_,
                "label(s) \"" + std::string(dangling_a) +
                    std::string(dangling_b) + "\"" +
                    " appear in one input only and not in the result; a free "
                    "reduction cannot be fused into a contraction");

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

inline void Equation::check_repeats() const {
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
}

inline Classification Equation::classify() const {
  check_repeats();

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

inline Classification Equation::classify_elementwise() const {
  check_repeats();
  if (operands_.size() != 2)
    throw Error(Errc::UnsupportedPattern, text_,
                "an elementwise operation takes two inputs, not " +
                    std::to_string(operands_.size()));

  const util::Index &a = operands_[0], &b = operands_[1], &d = result_;
  // Checked per input, as in classify_binary.
  const util::Index summed_a = a - d, summed_b = b - d;
  if (summed_a.rank() > 0 || summed_b.rank() > 0)
    throw Error(Errc::UnsupportedPattern, text_,
                "label(s) \"" + std::string(summed_a) + std::string(summed_b) +
                    "\" appear in an input but not in the result; an "
                    "elementwise operation sums over nothing, so use einsum");

  Classification c;
  c.kind = Lowering::Elementwise;
  c.batch = a & b;
  c.broadcast = d - (a & b);
  return c;
}

inline Classification Equation::classify_elementwise(
    std::error_code& ec) const noexcept {
  try {
    ec.clear();
    return classify_elementwise();
  } catch (const Error& e) {
    ec = make_error_code(e.code());
  } catch (...) {
    ec = make_error_code(Errc::UnsupportedPattern);
  }
  return Classification{};
}

}  // namespace katal

#endif  // KATAL_EQUATION_HPP_
