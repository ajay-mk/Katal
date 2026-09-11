#ifndef KATAL_ERROR_HPP_
#define KATAL_ERROR_HPP_

#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace katal {

/// Classified so a caller can react rather than only report.
enum class Errc {
  /// The equation is not well formed: an empty operand, a stray arrow, or a
  /// label containing one of the characters that separate operands.
  ParseError = 1,
  /// A label count disagrees with an extent count, or a rank exceeds
  /// `katal::max_rank`.
  RankMismatch,
  /// A label carries two different extents across the operands.
  ExtentMismatch,
  /// Well formed and correctly shaped, but not something Katal lowers: a
  /// free
  /// reduction beside a contraction, a broadcast, a diagonal, or more operands
  /// than the current lowering handles.
  UnsupportedPattern,
  /// The requested compute type cannot represent the operands, or alpha and
  /// beta cannot be expressed in it.
  DtypeMismatch,
  /// The workspace handed to `Plan::execute` is smaller than
  /// `Plan::workspace_size`.
  WorkspaceTooSmall,
  /// cuTENSOR returned a failing status; the message carries its own text.
  CutensorError,
  /// A CUDA runtime call failed.
  CudaError,
  /// A file could not be read or written, or held something other than what
  /// was asked for.
  IoError,
};

inline const std::error_category& error_category() noexcept {
  class Category final : public std::error_category {
   public:
    const char* name() const noexcept override { return "katal"; }

    std::string message(int value) const override {
      switch (static_cast<Errc>(value)) {
        case Errc::ParseError:
          return "malformed equation";
        case Errc::RankMismatch:
          return "rank mismatch";
        case Errc::ExtentMismatch:
          return "extent mismatch";
        case Errc::UnsupportedPattern:
          return "unsupported contraction pattern";
        case Errc::DtypeMismatch:
          return "data type mismatch";
        case Errc::WorkspaceTooSmall:
          return "workspace too small";
        case Errc::CutensorError:
          return "cuTENSOR error";
        case Errc::CudaError:
          return "CUDA error";
        case Errc::IoError:
          return "file I/O error";
      }
      return "unrecognized Katal error";
    }
  };

  static const Category instance;
  return instance;
}

inline std::error_code make_error_code(Errc e) noexcept {
  return {static_cast<int>(e), error_category()};
}

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

template <>
struct std::is_error_code_enum<katal::Errc> : std::true_type {};

#endif  // KATAL_ERROR_HPP_
