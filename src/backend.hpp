#ifndef KATAL_SRC_BACKEND_HPP_
#define KATAL_SRC_BACKEND_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <katal/context.hpp>
#include <katal/options.hpp>
#include <katal/plan.hpp>
#include <katal/scalar.hpp>
#include <katal/tensor_ref.hpp>

/// The seam between Katal's front end and the library that runs the work.
///
/// The front end parses and classifies an equation, checks shapes, and
/// lowers it to an Operation: modes, extents, strides and types, in the terms
/// every tensor library shares. A Backend turns an Operation into a Kernel
/// and runs it. The Operation, Kernel and Backend types name no library, so a
/// second backend implements the two classes and leaves the lowering alone;
/// choosing it is a change to Context::Impl, which builds cuTENSOR's through
/// make_cutensor_backend. Devices are still CUDA devices.
namespace katal::backend {

/// What an Operation computes:
/// - Contraction, `d = alpha * A.B + beta * c`, summing the shared modes the
///   output drops.
/// - Reduction, `d = alpha * reduce(A) + beta * c` over the modes the output
///   drops, reducing with `Operation::binary`. A reduction over no modes is a
///   permutation or a copy.
/// - Elementwise, `d = op(alpha * A, beta * B) + gamma * c`, with an input
///   broadcast along any output mode it lacks.
/// - Permutation, `d = alpha * A`, with A permuted and broadcast to the
///   output's modes and converted to its type. No beta, no workspace.
///
/// Every input is read through its Operand's unary `op`.
enum class OpKind { Contraction, Reduction, Elementwise, Permutation };

/// One tensor as the backend sees it: modes in descriptor order, with the
/// extents and strides (in elements) permuted to match. A mode id is the same
/// number in every operand that shares the label.
struct Operand {
  std::vector<std::int32_t> modes;
  Extents extent;
  Extents stride;
  Dtype type = Dtype::F64;
  /// The largest power of two, up to 256, that the data pointer is planned to
  /// be aligned to.
  std::uint32_t alignment = 256;
  /// Applied to each element as an input is read. Identity for the output.
  UnaryOp op = UnaryOp::Identity;
};

/// A fully checked request. `c` shares `output`'s layout, so beta scales the
/// output in place.
struct Operation {
  OpKind kind = OpKind::Contraction;
  /// A, and B for a contraction.
  std::vector<Operand> inputs;
  Operand output;
  /// The accumulation type the front end chose.
  Dtype compute = Dtype::F64;
  /// How an elementwise operation combines its inputs, or how a reduction
  /// reduces; unused otherwise.
  BinaryOp binary = BinaryOp::Add;
  /// The equation text, which every error a backend raises quotes.
  std::string equation;
};

/// A planned Operation, ready to run many times. Const and stream-safe, so
/// one Kernel may be shared across threads.
class Kernel {
 public:
  Kernel() = default;
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;
  virtual ~Kernel() = default;

  /// The scratch `run` needs, which may be zero.
  virtual std::uint64_t workspace_size() const noexcept = 0;

  /// Queues the work on `stream`. `scalars` holds the operation's factors in
  /// the order OpKind writes them: alpha and beta, alpha, beta and gamma for
  /// an elementwise one, or alpha alone for a permutation. `inputs` holds one
  /// device pointer per Operation input, in the same order. The front end has
  /// already checked the device, the alignments and the workspace size.
  virtual void run(std::span<const Scalar> scalars,
                   std::span<const void* const> inputs, const void* c, void* d,
                   Workspace workspace, StreamRef stream) const = 0;
};

/// One library on one device. A Context owns exactly one, and every Kernel it
/// plans borrows it, so it must outlive them.
class Backend {
 public:
  Backend() = default;
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;
  virtual ~Backend() = default;

  /// The device this backend was created on.
  virtual int device() const noexcept = 0;

  /// Throws `Errc::CudaError` naming `owner` ("plan", "context") if another
  /// device is current. `equation` may be empty.
  virtual void check_device(std::string_view equation,
                            std::string_view owner) const = 0;

  /// Selects a kernel for `operation`. Throws `katal::Error` if the library
  /// cannot run it.
  virtual std::unique_ptr<Kernel> plan(const Operation& operation,
                                       const PlanOptions& options) const = 0;
};

/// The cuTENSOR backend on the current CUDA device.
std::unique_ptr<Backend> make_cutensor_backend(const Config& config);

}  // namespace katal::backend

#endif  // KATAL_SRC_BACKEND_HPP_
