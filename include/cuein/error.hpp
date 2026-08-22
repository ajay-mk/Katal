#ifndef CUEIN_ERROR_HPP_
#define CUEIN_ERROR_HPP_

#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace cuein {

/// Classified so a caller can react rather than only report.
enum class Errc {
  ParseError = 1,
  RankMismatch,
  ExtentMismatch,
  UnsupportedPattern,
  DtypeMismatch,
  WorkspaceTooSmall,
  CutensorError,
  CudaError,
};

inline const std::error_category& error_category() noexcept {
  class Category final : public std::error_category {
   public:
    const char* name() const noexcept override { return "cuein"; }

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
      }
      return "unrecognized cuEin error";
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

  Errc code() const noexcept { return code_; }
  const std::string& equation() const noexcept { return equation_; }

 private:
  static std::string compose(std::string_view equation_text,
                             std::string_view detail) {
    std::string s = "cuein: ";
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

}  // namespace cuein

template <>
struct std::is_error_code_enum<cuein::Errc> : std::true_type {};

#endif  // CUEIN_ERROR_HPP_
