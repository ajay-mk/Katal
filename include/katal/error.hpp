#ifndef KATAL_ERROR_HPP_
#define KATAL_ERROR_HPP_

#include <stdexcept>
#include <string>
#include <string_view>

namespace katal {

/// Classified so a caller can react rather than only report.
enum class Errc {
  /// The equation is not well formed: an empty operand, a stray arrow, or a
  /// label containing one of the characters that separate operands.
  ParseError = 1,
  /// A label count disagrees with an extent count, or a rank exceeds
  /// `katal::max_rank`.
  RankMismatch,
  /// A label carries two different extents across the operands, an extent is
  /// not positive, or a shape is too large to address.
  ExtentMismatch,
  /// Well formed and correctly shaped, but not something Katal lowers: a
  /// broadcast in einsum, a repeated result label, more
  /// operands than the current lowering handles, or a cuTENSOR operation
  /// reported as not supported. Also a pointer handed to `Plan::execute` that
  /// is less aligned than the plan was built for, an elementwise or
  /// permutation input passed as the output, a permutation given a beta, an
  /// output carrying a unary operator, an output whose strides address one
  /// element twice, an execute overload that does not match the plan, or a
  /// workspace less than 256-byte aligned for a plan that keeps a reduced
  /// input in it.
  UnsupportedPattern,
  /// An operand or scalar representation is unsupported. Scalar narrowing and
  /// discarding an imaginary part for real computation do not raise this error.
  DtypeMismatch,
  /// The workspace handed to `Plan::execute` is smaller than
  /// `Plan::workspace_size`, or cuTENSOR reported too little workspace.
  WorkspaceTooSmall,
  /// cuTENSOR returned a failing status; the message carries its own text.
  CutensorError,
  /// A CUDA runtime call failed.
  CudaError,
  /// The process-wide context was misused: a second `initialize`, a `Runtime`
  /// while the default context exists, or `finalize` while a `Runtime` owns
  /// it. Also a failed `KATAL_ASSERT` when Katal was configured to throw on
  /// one, such as a moved-from Context or Plan being used.
  InvalidState,
};

/// Carries the equation text alongside the failure, which is the most useful
/// thing to have in hand when a contraction is rejected.
class Error : public std::runtime_error {
 public:
  Error(Errc code, std::string_view equation_text, std::string_view detail)
      : std::runtime_error(compose(equation_text, detail)),
        code_(code),
        equation_(equation_text) {}

  /// The classification a caller can branch on, as opposed to the message,
  /// which is for a human.
  Errc code() const noexcept { return code_; }

  /// The equation that failed, empty when the failure had no equation in hand.
  const std::string& equation() const noexcept { return equation_; }

 private:
  static std::string compose(std::string_view equation_text,
                             std::string_view detail) {
    std::string s = "katal: ";
    if (!equation_text.empty()) {
      s += "in \"";
      s += equation_text;
      s += "\": ";
    }
    s += detail;
    return s;
  }

  Errc code_;
  std::string equation_;
};

}  // namespace katal

#endif  // KATAL_ERROR_HPP_
