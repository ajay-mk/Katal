#ifndef KATAL_SRC_LOG_HPP_
#define KATAL_SRC_LOG_HPP_

#if KATAL_ENABLE_LOGGING
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <syncstream>

#include <enumstr.hpp>

#include <katal/options.hpp>
#include <katal/plan.hpp>

namespace katal::util {

inline std::string log_quote(std::string_view text) {
  std::string result = "\"";
  for (const unsigned char c : text) {
    if (c < 32 || c >= 127) {
      constexpr char hex[] = "0123456789abcdef";
      result += "\\x";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      if (c == '"' || c == '\\') result += '\\';
      result += static_cast<char>(c);
    }
  }
  return result + '"';
}

inline std::string log_tensor(const ConstTensorRef& tensor) {
  std::ostringstream output;
  output << enumstr::to_string(tensor.type) << '[';
  for (std::size_t i = 0; i < tensor.extent.size(); ++i) {
    if (i != 0) output << ',';
    output << tensor.extent[i];
  }
  output << "] strides=[";
  for (std::size_t i = 0; i < tensor.stride.size(); ++i) {
    if (i != 0) output << ',';
    output << tensor.stride[i];
  }
  output << "] layout="
         << (tensor.layout == Layout::ColumnMajor ? "column" : "row")
         << " alignment=" << tensor.alignment;
  if (tensor.op != UnaryOp::Identity)
    output << " op=" << enumstr::to_string(tensor.op);
  return output.str();
}

inline std::string log_options(const PlanOptions& options) {
  std::ostringstream output;
  output << "algo=" << enumstr::to_string(options.algo) << " compute="
         << (options.compute ? enumstr::to_string(*options.compute)
                             : std::string_view("output"))
         << " workspace_preference=" << enumstr::to_string(options.workspace)
         << " workspace_limit=" << options.workspace_limit
         << " kernel_rank=" << options.kernel_rank.value_or(-1)
         << " jit=" << options.jit
         << " precision=" << enumstr::to_string(options.precision);
  return output.str();
}

template <class Write>
void log(Write write) noexcept {
  try {
    std::osyncstream output(std::cerr);
    output << "[katal] ";
    write(output);
    output << '\n';
  } catch (...) {
    // Diagnostics must not change an operation's success or failure.
  }
}

}  // namespace katal::util

#define KATAL_LOG(message) \
  ::katal::util::log(      \
      [&](auto& katal_log_output) { katal_log_output << message; })
#else
// Discard the arguments too: disabled logging must not format or time work.
#define KATAL_LOG(message) ((void)0)
#endif

#endif  // KATAL_SRC_LOG_HPP_
