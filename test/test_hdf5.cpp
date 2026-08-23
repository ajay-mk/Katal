#include <cstdint>
#include <filesystem>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <cuein/error.hpp>
#include <cuein/io/hdf5.hpp>

namespace {

/// A path in the system temporary directory, removed when the test ends.
class TempFile {
 public:
  explicit TempFile(const char* name)
      : path_(std::filesystem::temp_directory_path() / name) {
    std::filesystem::remove(path_);
  }
  ~TempFile() { std::filesystem::remove(path_); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace

TEST_CASE("an array survives a write and a read", "[hdf5]") {
  const TempFile file("cuein_roundtrip.h5");

  // Asymmetric, so a shape read back transposed would show up.
  const std::vector<std::int64_t> extent{2, 3, 4};
  std::vector<double> data(24);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = 1.0 + static_cast<double>(i) / 8.0;

  cuein::io::write(file.path(), "block", data, extent);
  const cuein::io::Array read = cuein::io::read(file.path(), "block");

  REQUIRE(read.extent == extent);
  REQUIRE(read.elements() == 24);
  REQUIRE(read.data == data);
}

TEST_CASE("two datasets can share a file", "[hdf5]") {
  const TempFile file("cuein_two.h5");
  const std::vector<std::int64_t> extent{2, 2};

  cuein::io::write(file.path(), "first", std::vector<double>{1, 2, 3, 4},
                   extent);
  cuein::io::write(file.path(), "second", std::vector<double>{5, 6, 7, 8},
                   extent);

  REQUIRE(cuein::io::read(file.path(), "first").data[0] == 1.0);
  REQUIRE(cuein::io::read(file.path(), "second").data[3] == 8.0);
}

TEST_CASE("hdf5 failures say what was wrong", "[hdf5]") {
  const TempFile file("cuein_missing.h5");

  SECTION("a file that is not there") {
    REQUIRE_THROWS_AS(cuein::io::read(file.path(), "block"), cuein::Error);
  }

  SECTION("a dataset that is not there") {
    cuein::io::write(file.path(), "present", std::vector<double>{1.0}, {{1}});
    REQUIRE_THROWS_AS(cuein::io::read(file.path(), "absent"), cuein::Error);
    REQUIRE_THROWS_AS(cuein::io::read_attribute(file.path(), "absent"),
                      cuein::Error);
  }

  SECTION("extents that disagree with the data") {
    const std::vector<std::int64_t> extent{3, 3};
    REQUIRE_THROWS_AS(cuein::io::write(file.path(), "wrong",
                                       std::vector<double>{1.0}, extent),
                      cuein::Error);
  }
}
