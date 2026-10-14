#ifndef KATAL_TEST_ENUM_NAMES_HPP_
#define KATAL_TEST_ENUM_NAMES_HPP_

#include <string>
#include <string_view>
#include <type_traits>

#include <catch2/catch_tostring.hpp>
#include <enumstr.hpp>

/// Catch2 prints a failed comparison of scoped enums by name, "C32 == F32",
/// rather than by value. A value enumstr cannot name falls back to its number.
/// Unscoped enums such as cudaError_t keep Catch2's own printing.
template <class E>
struct Catch::StringMaker<
    E, std::enable_if_t<std::is_enum_v<E> &&
                        !std::is_convertible_v<E, std::underlying_type_t<E>>>> {
  static std::string convert(E value) {
    const std::string_view name = enumstr::to_string_view(value);
    if (name == "<unknown>")
      return std::to_string(static_cast<std::underlying_type_t<E>>(value));
    return std::string(name);
  }
};

#endif  // KATAL_TEST_ENUM_NAMES_HPP_
