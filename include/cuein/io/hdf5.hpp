#ifndef CUEIN_IO_HDF5_HPP_
#define CUEIN_IO_HDF5_HPP_

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cuein::io {

/// One dense f64 array read from a file, with its shape. Double precision only:
/// the data this exists to move is coupled-cluster amplitudes and integrals,
/// which are never anything else.
struct Array {
  std::vector<std::int64_t> extent;
  std::vector<double> data;

  std::int64_t elements() const noexcept {
    std::int64_t n = 1;
    for (std::int64_t e : extent) n *= e;
    return n;
  }
};

/// Reads a named dataset. Throws `Error` with `Errc::IoError` if the file or
/// the dataset is missing, or if the dataset is not f64.
Array read(const std::filesystem::path& file, std::string_view dataset);

/// Reads a string attribute from the file root. Used to carry the einsum
/// equation alongside the tensors, so a reader never has to assume an index
/// convention.
std::string read_attribute(const std::filesystem::path& file,
                           std::string_view name);

/// Reads a scalar attribute from the file root.
double read_scalar(const std::filesystem::path& file, std::string_view name);

/// Writes a dense f64 array, creating or truncating the file on the first call
/// and appending on later ones.
void write(const std::filesystem::path& file, std::string_view dataset,
           std::span<const double> data, std::span<const std::int64_t> extent);

}  // namespace cuein::io

#endif  // CUEIN_IO_HDF5_HPP_
