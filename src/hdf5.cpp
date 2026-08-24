#include <utility>

#include <hdf5.h>

#include <deviceein/error.hpp>
#include <deviceein/io/hdf5.hpp>

namespace deviceein::io {
namespace {

/// Closes an HDF5 identifier on scope exit. The C API hands out integer
/// handles, so RAII has to be applied by hand.
class Handle {
 public:
  Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
  ~Handle() {
    if (id_ >= 0) close_(id_);
  }
  Handle(Handle&& other) noexcept : id_(other.id_), close_(other.close_) {
    other.id_ = -1;
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle& operator=(Handle&&) = delete;

  hid_t get() const noexcept { return id_; }
  bool valid() const noexcept { return id_ >= 0; }

 private:
  hid_t id_ = -1;
  herr_t (*close_)(hid_t) = nullptr;
};

Handle open_file(const std::filesystem::path& file) {
  Handle handle(H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!handle.valid())
    throw Error(Errc::IoError, file.string(), "cannot open the file");
  return handle;
}

}  // namespace

Array read(const std::filesystem::path& file, std::string_view dataset) {
  const Handle id = open_file(file);
  const std::string name(dataset);

  const Handle set(H5Dopen2(id.get(), name.c_str(), H5P_DEFAULT), H5Dclose);
  if (!set.valid())
    throw Error(Errc::IoError, file.string(),
                "no dataset named \"" + name + "\"");

  const Handle type(H5Dget_type(set.get()), H5Tclose);
  if (H5Tget_class(type.get()) != H5T_FLOAT ||
      H5Tget_size(type.get()) != sizeof(double))
    throw Error(Errc::IoError, file.string(),
                "dataset \"" + name + "\" is not double precision");

  const Handle space(H5Dget_space(set.get()), H5Sclose);
  const int rank = H5Sget_simple_extent_ndims(space.get());
  if (rank < 0)
    throw Error(Errc::IoError, file.string(),
                "dataset \"" + name + "\" has no shape");

  std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
  H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr);

  Array array;
  for (hsize_t d : dims) array.extent.push_back(static_cast<std::int64_t>(d));
  array.data.resize(static_cast<std::size_t>(array.elements()));

  if (H5Dread(set.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              array.data.data()) < 0)
    throw Error(Errc::IoError, file.string(),
                "cannot read dataset \"" + name + "\"");

  return array;
}

std::string read_attribute(const std::filesystem::path& file,
                           std::string_view name) {
  const Handle id = open_file(file);
  const std::string key(name);

  const Handle attribute(H5Aopen(id.get(), key.c_str(), H5P_DEFAULT), H5Aclose);
  if (!attribute.valid())
    throw Error(Errc::IoError, file.string(),
                "no attribute named \"" + key + "\"");

  const Handle stored(H5Aget_type(attribute.get()), H5Tclose);
  if (H5Tget_class(stored.get()) != H5T_STRING)
    throw Error(Errc::IoError, file.string(),
                "attribute \"" + key + "\" is not a string");

  // Reading through a variable-length memory type covers both the fixed and
  // variable length cases the writer might have used.
  const Handle memory(H5Tcopy(H5T_C_S1), H5Tclose);
  H5Tset_size(memory.get(), H5T_VARIABLE);
  H5Tset_cset(memory.get(), H5Tget_cset(stored.get()));

  char* text = nullptr;
  if (H5Aread(attribute.get(), memory.get(), &text) < 0 || text == nullptr)
    throw Error(Errc::IoError, file.string(),
                "cannot read attribute \"" + key + "\"");

  std::string value(text);
  H5free_memory(text);
  return value;
}

double read_scalar(const std::filesystem::path& file, std::string_view name) {
  const Handle id = open_file(file);
  const std::string key(name);

  const Handle attribute(H5Aopen(id.get(), key.c_str(), H5P_DEFAULT), H5Aclose);
  if (!attribute.valid())
    throw Error(Errc::IoError, file.string(),
                "no attribute named \"" + key + "\"");

  double value = 0.0;
  if (H5Aread(attribute.get(), H5T_NATIVE_DOUBLE, &value) < 0)
    throw Error(Errc::IoError, file.string(),
                "cannot read attribute \"" + key + "\" as a number");
  return value;
}

void write(const std::filesystem::path& file, std::string_view dataset,
           std::span<const double> data, std::span<const std::int64_t> extent) {
  const bool exists = std::filesystem::exists(file);
  const Handle id(
      exists ? H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT)
             : H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
      H5Fclose);
  if (!id.valid())
    throw Error(Errc::IoError, file.string(),
                "cannot open the file for writing");

  std::vector<hsize_t> dims;
  std::int64_t elements = 1;
  for (std::int64_t e : extent) {
    dims.push_back(static_cast<hsize_t>(e));
    elements *= e;
  }
  if (elements != static_cast<std::int64_t>(data.size()))
    throw Error(Errc::IoError, file.string(),
                "the extents describe a different number of elements than the "
                "data holds");

  const Handle space(
      H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr),
      H5Sclose);
  const std::string name(dataset);
  const Handle set(
      H5Dcreate2(id.get(), name.c_str(), H5T_IEEE_F64LE, space.get(),
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose);
  if (!set.valid())
    throw Error(Errc::IoError, file.string(),
                "cannot create dataset \"" + name + "\"");

  if (H5Dwrite(set.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               data.data()) < 0)
    throw Error(Errc::IoError, file.string(),
                "cannot write dataset \"" + name + "\"");
}

}  // namespace deviceein::io
