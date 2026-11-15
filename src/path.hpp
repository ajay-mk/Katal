#ifndef KATAL_SRC_PATH_HPP_
#define KATAL_SRC_PATH_HPP_

// Lowers an einsum over three or more operands to a chain of two-operand
// contractions along a given order, and lays out the intermediates they pass
// between them. Host-only bookkeeping: nothing here plans a kernel or touches a
// device, so the result can be checked without a GPU. A composite plan built
// on it, and slicing to keep the intermediates under a memory limit, come
// later.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <katal/equation.hpp>
#include <katal/error.hpp>
#include <katal/tensor_ref.hpp>
#include <katal/util/index.hpp>

namespace katal::detail {

/// One pairwise contraction in opt_einsum's path format: two positions in the
/// current operand list. Both are removed and their result is appended, so
/// `{{0, 1}, {0, 1}}` on three operands contracts the first two, then the
/// third with that intermediate.
using PathPair = std::pair<std::size_t, std::size_t>;

/// One binary contraction of a lowered path.
struct PathStep {
  /// Operand ids: `0 .. n-1` are the einsum's inputs, `n + k` is the
  /// intermediate step `k` produces.
  std::size_t a = 0, b = 0;
  /// Labels of what the step produces. The last step produces the einsum's
  /// result, in its written order.
  util::Index labels;
  /// The step as a two-operand equation, e.g. `"klcd,ijcd->klij"`.
  std::string equation;
  /// Multiply-adds: the product of the extents of every label in either
  /// input. The FLOP count of a contraction is about twice this.
  double multiply_adds = 0;
  /// Bytes of the intermediate, dense. 0 for the last step, which writes the
  /// caller's output rather than workspace.
  std::uint64_t bytes = 0;
  /// Where the intermediate starts in the workspace, a multiple of
  /// `path_alignment`. 0 and meaningless for the last step.
  std::uint64_t offset = 0;
};

/// A lowered path: the steps in execution order and the workspace their
/// intermediates need.
struct Path {
  std::vector<PathStep> steps;
  /// Bytes from the start of the workspace to the end of the furthest
  /// intermediate. Intermediates whose lifetimes do not overlap share bytes,
  /// so this is the peak a memory limit is compared against. Kernel scratch
  /// is not included; it depends on the backend's kernel choice.
  std::uint64_t workspace = 0;
  /// Sum of every step's multiply-adds.
  double multiply_adds = 0;
};

/// Each intermediate starts on this boundary, which is what cuTENSOR asks of
/// its operands for the fastest kernels.
inline constexpr std::uint64_t path_alignment = 256;

inline std::uint64_t path_align_up(std::uint64_t n, std::uint64_t to,
                                   std::string_view text) {
  if (n > std::numeric_limits<std::uint64_t>::max() - (to - 1))
    throw Error(Errc::ExtentMismatch, text,
                "an intermediate is too large to address");
  return (n + to - 1) / to * to;
}

/// Bytes of a dense tensor with these labels. Throws on overflow.
inline std::uint64_t path_bytes(const util::Index& labels,
                                const std::array<std::int64_t, 256>& extent,
                                std::size_t element, std::string_view text) {
  std::uint64_t n = element;
  for (const util::Label l : labels) {
    const auto e =
        static_cast<std::uint64_t>(extent[static_cast<unsigned char>(l)]);
    if (n > std::numeric_limits<std::uint64_t>::max() / e)
      throw Error(Errc::ExtentMismatch, text,
                  "the intermediate \"" + std::string(labels) +
                      "\" is too large to address");
    n *= e;
  }
  return n;
}

/// Lowers `equation` along `path` into binary steps over intermediates whose
/// elements are `type`.
///
/// An intermediate keeps every label of its two inputs that a later operand
/// or the result still needs, in the order they first appear in its inputs;
/// every other label is summed in the step that produces it. Intermediates
/// are placed first-fit by lifetime: one is live from the step that writes it
/// through the step that reads it, and two whose lifetimes overlap never share
/// bytes.
///
/// `extents` holds one entry per operand, in the order the equation writes
/// them. Throws `Errc::RankMismatch` or `Errc::ExtentMismatch` for extents
/// that do not fit the equation, `Errc::UnsupportedPattern` for fewer than two
/// operands or a path that is not `n - 1` pairs of distinct, in-range
/// positions, and whatever classifying a step throws, naming that step.
inline Path lower_path(const Equation& equation,
                       std::span<const std::span<const std::int64_t>> extents,
                       std::span<const PathPair> path, Dtype type) {
  const std::string_view text = equation.text();
  const std::size_t n = equation.n_operands();

  if (n < 2)
    throw Error(Errc::UnsupportedPattern, text,
                "a contraction path needs at least two operands");
  if (extents.size() != n)
    throw Error(Errc::RankMismatch, text,
                "the equation has " + std::to_string(n) + " operands but " +
                    std::to_string(extents.size()) + " extents were given");
  if (path.size() != n - 1)
    throw Error(Errc::UnsupportedPattern, text,
                "a path over " + std::to_string(n) + " operands has " +
                    std::to_string(n - 1) + " steps, not " +
                    std::to_string(path.size()));

  // One extent per label, agreed by every operand that carries it.
  std::array<std::int64_t, 256> extent{};
  for (std::size_t i = 0; i < n; ++i) {
    const util::Index& labels = equation.operand(i);
    if (extents[i].size() != labels.rank())
      throw Error(Errc::RankMismatch, text,
                  "operand " + std::to_string(i) + " \"" + std::string(labels) +
                      "\" has " + std::to_string(labels.rank()) +
                      " labels but " + std::to_string(extents[i].size()) +
                      " extents");
    for (std::size_t m = 0; m < labels.rank(); ++m) {
      const std::int64_t e = extents[i][m];
      if (e <= 0)
        throw Error(Errc::ExtentMismatch, text,
                    "tensor extents must be positive");
      std::int64_t& known = extent[static_cast<unsigned char>(labels[m])];
      if (known != 0 && known != e)
        throw Error(Errc::ExtentMismatch, text,
                    std::string("label '") + labels[m] + "' has extents " +
                        std::to_string(known) + " and " + std::to_string(e));
      known = e;
    }
  }

  // The operands still waiting, as ids, and each id's labels.
  std::vector<std::size_t> pending(n);
  std::vector<util::Index> labels_of;
  for (std::size_t i = 0; i < n; ++i) {
    pending[i] = i;
    labels_of.push_back(equation.operand(i));
  }

  const std::size_t element = util::size_of(type);
  Path out;
  out.steps.reserve(n - 1);
  for (std::size_t k = 0; k < path.size(); ++k) {
    const auto [p, q] = path[k];
    if (p == q || p >= pending.size() || q >= pending.size())
      throw Error(Errc::UnsupportedPattern, text,
                  "path step " + std::to_string(k) + " is (" +
                      std::to_string(p) + ", " + std::to_string(q) +
                      "), which is not two distinct positions among " +
                      std::to_string(pending.size()) + " operands");

    PathStep step;
    step.a = pending[p];
    step.b = pending[q];
    pending.erase(pending.begin() +
                  static_cast<std::ptrdiff_t>(std::max(p, q)));
    pending.erase(pending.begin() +
                  static_cast<std::ptrdiff_t>(std::min(p, q)));

    const util::Index& a = labels_of[step.a];
    const util::Index& b = labels_of[step.b];
    const bool last = k + 1 == path.size();
    // Label sets are tables over every byte, not Index: the labels still
    // pending, or both inputs of a step together, can outnumber max_rank even
    // when no tensor does.
    std::array<bool, 256> in_step{};
    for (const util::Label l : a) in_step[static_cast<unsigned char>(l)] = true;
    for (const util::Label l : b) in_step[static_cast<unsigned char>(l)] = true;
    if (last) {
      step.labels = equation.result();
    } else {
      std::array<bool, 256> needed{};
      for (const util::Label l : equation.result())
        needed[static_cast<unsigned char>(l)] = true;
      for (const std::size_t id : pending)
        for (const util::Label l : labels_of[id])
          needed[static_cast<unsigned char>(l)] = true;
      for (const util::Index* input : {&a, &b})
        for (const util::Label l : *input)
          if (needed[static_cast<unsigned char>(l)] && !step.labels.contains(l))
            step.labels.push_back(l);
    }
    step.equation =
        std::string(a) + "," + std::string(b) + "->" + std::string(step.labels);

    // Each step must be a contraction Katal lowers on its own; say which
    // step failed as well as the einsum it came from.
    try {
      (void)Equation::parse(step.equation).classify();
    } catch (const Error& e) {
      throw Error(e.code(), text,
                  "path step " + std::to_string(k) + " \"" + step.equation +
                      "\": " + e.what());
    }

    step.multiply_adds = 1;
    for (std::size_t c = 0; c < in_step.size(); ++c)
      if (in_step[c]) step.multiply_adds *= static_cast<double>(extent[c]);
    out.multiply_adds += step.multiply_adds;
    if (!last) step.bytes = path_bytes(step.labels, extent, element, text);

    labels_of.push_back(step.labels);
    pending.push_back(n + k);
    out.steps.push_back(std::move(step));
  }

  // Lifetimes: intermediate n + k is written by step k and read by the step
  // that consumes it, and is live over both, since a step's output must not
  // overlap its inputs.
  std::vector<std::size_t> read_by(out.steps.size(), 0);
  for (std::size_t s = 0; s < out.steps.size(); ++s)
    for (const std::size_t id : {out.steps[s].a, out.steps[s].b})
      if (id >= n) read_by[id - n] = s;

  for (std::size_t k = 0; k + 1 < out.steps.size(); ++k) {
    // Byte ranges taken by intermediates alive at the same time as this one.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> taken;
    for (std::size_t j = 0; j < k; ++j)
      if (read_by[j] >= k)
        taken.emplace_back(out.steps[j].offset,
                           out.steps[j].offset + out.steps[j].bytes);
    std::sort(taken.begin(), taken.end());

    PathStep& step = out.steps[k];
    std::uint64_t offset = 0;
    for (const auto& [begin, end] : taken) {
      if (offset + step.bytes <= begin) break;
      offset = std::max(offset, path_align_up(end, path_alignment, text));
    }
    if (offset > std::numeric_limits<std::uint64_t>::max() - step.bytes)
      throw Error(Errc::ExtentMismatch, text,
                  "the intermediates are too large to address");
    step.offset = offset;
    out.workspace = std::max(out.workspace, offset + step.bytes);
  }
  return out;
}

}  // namespace katal::detail

#endif  // KATAL_SRC_PATH_HPP_
